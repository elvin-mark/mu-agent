"""Unified multi-provider LLM API wrapper (pi-ai equivalent)."""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from .types import Message, Role, StreamChunk, Usage


class BaseLLMProvider:
    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-4o",
    ):
        import openai

        api_key = api_key or os.environ.get("OPENAI_API_KEY") or "lm-studio"
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        formatted_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                formatted_messages.append(
                    {"role": "system", "content": msg.content or ""}
                )
            elif msg.role == Role.USER:
                formatted_messages.append(
                    {"role": "user", "content": msg.content or ""}
                )
            elif msg.role == Role.ASSISTANT:
                item: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    item["content"] = msg.content
                if msg.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tc.id or f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else tc.arguments,
                            },
                        }
                        for idx, tc in enumerate(msg.tool_calls)
                    ]
                formatted_messages.append(item)

            elif msg.role == Role.TOOL and msg.tool_result:
                formatted_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_result.tool_call_id or "call_0",
                        "name": msg.tool_result.name,
                        "content": msg.tool_result.output,
                    }
                )

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": formatted_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            formatted_tools = []
            for t in tools:
                params = t.get("parameters", {})
                if not params:
                    params = {"type": "object", "properties": {}}
                elif "type" not in params:
                    params["type"] = "object"

                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": params,
                        },
                    }
                )
            kwargs["tools"] = formatted_tools

        import time

        start_time = time.time()

        stream = await self.client.chat.completions.create(**kwargs)

        current_tc_id = ""
        current_tc_name = ""
        current_tc_args = ""
        has_tool_call = False
        completion_text_buf = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

            # Check for usage object from OpenAI streaming
            if hasattr(chunk, "usage") and chunk.usage:
                u = chunk.usage
                yield StreamChunk(
                    usage=Usage(
                        prompt_tokens=u.prompt_tokens or 0,
                        completion_tokens=u.completion_tokens or 0,
                        total_tokens=u.total_tokens or 0,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )

            if not delta:
                continue

            if delta.content:
                completion_text_buf += delta.content
                yield StreamChunk(delta_content=delta.content)

            if delta.tool_calls:
                has_tool_call = True
                for tc_delta in delta.tool_calls:
                    if tc_delta.id:
                        current_tc_id = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        current_tc_name = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        current_tc_args += tc_delta.function.arguments

            if finish_reason in ("tool_calls", "function_call") or (
                finish_reason == "stop" and has_tool_call
            ):
                if not current_tc_id:
                    current_tc_id = f"call_ollama_{os.urandom(4).hex()}"
                try:
                    args_dict = json.loads(current_tc_args) if current_tc_args else {}
                except Exception:
                    args_dict = {"raw": current_tc_args}
                yield StreamChunk(
                    delta_tool_call={
                        "id": current_tc_id,
                        "name": current_tc_name,
                        "arguments": args_dict,
                    },
                    finish_reason="tool_calls",
                )
                has_tool_call = False
            elif finish_reason:
                yield StreamChunk(finish_reason=finish_reason)

        # Fallback tiktoken calculation if provider did not emit usage object (e.g. Ollama/LMStudio)
        import tiktoken

        try:
            enc = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = sum(
                len(enc.encode(m.get("content", "") or "")) for m in formatted_messages
            )
            completion_tokens = len(enc.encode(completion_text_buf or current_tc_args))
        except Exception:
            prompt_tokens = sum(
                len(m.get("content", "") or "") // 4 for m in formatted_messages
            )
            completion_tokens = len(completion_text_buf or current_tc_args) // 4

        yield StreamChunk(
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=(time.time() - start_time) * 1000,
            )
        )


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-3-5-sonnet-20241022",
    ):
        import anthropic

        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.default_model = default_model

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt = ""
        formatted_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt += (msg.content or "") + "\n"
            elif msg.role == Role.USER:
                formatted_messages.append(
                    {"role": "user", "content": msg.content or ""}
                )
            elif msg.role == Role.ASSISTANT:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                formatted_messages.append(
                    {"role": "assistant", "content": content or (msg.content or "")}
                )
            elif msg.role == Role.TOOL and msg.tool_result:
                formatted_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_result.tool_call_id,
                                "content": msg.tool_result.output,
                                "is_error": msg.tool_result.is_error,
                            }
                        ],
                    }
                )

        anthropic_tools = []
        if tools:
            for t in tools:
                anthropic_tools.append(
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": formatted_messages,
            "max_tokens": 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt.strip()
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "text":
                    yield StreamChunk(delta_content=event.text)
                elif event.type == "tool_use":
                    yield StreamChunk(
                        delta_tool_call={
                            "id": event.id,
                            "name": event.name,
                            "arguments": event.input,
                        },
                        finish_reason="tool_calls",
                    )


def get_provider(
    provider_name: str = "openai", base_url: str | None = None, **kwargs
) -> BaseLLMProvider:
    if provider_name == "anthropic":
        return AnthropicProvider(**kwargs)
    elif provider_name == "ollama":
        url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAIProvider(base_url=url, **kwargs)

    # Defaults to OpenAI or local OpenAI-compatible endpoint
    return OpenAIProvider(base_url=base_url, **kwargs)
