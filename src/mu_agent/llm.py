"""Unified multi-provider LLM API wrapper for mu-agent."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import tiktoken

from .types import Message, Role, StreamChunk, ToolCallDelta, Usage

logger = logging.getLogger(__name__)

# Cache tiktoken encoder once at module level — not on every request.
try:
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None


def _estimate_tokens_fallback(texts: list[str]) -> int:
    """Rough fallback: ~4 chars per token."""
    return sum(len(t) for t in texts) // 4


def _count_tokens(text: str) -> int:
    if _TOKEN_ENCODER is not None:
        try:
            return len(_TOKEN_ENCODER.encode(text))
        except Exception:
            pass
    return len(text) // 4


def _format_tools_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal tool schema list to OpenAI function-call format."""
    formatted: list[dict[str, Any]] = []
    for t in tools:
        params = t.get("parameters", {})
        if not params:
            params = {"type": "object", "properties": {}}
        elif "type" not in params:
            params = {"type": "object", **params}
        formatted.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return formatted


def _format_messages_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Serialize Message list into OpenAI chat format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == Role.SYSTEM:
            result.append({"role": "system", "content": msg.content or ""})
        elif msg.role == Role.USER:
            result.append({"role": "user", "content": msg.content or ""})
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
            result.append(item)
        elif msg.role == Role.TOOL and msg.tool_result:
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_result.tool_call_id or "call_0",
                    "name": msg.tool_result.name,
                    "content": msg.tool_result.output,
                }
            )
    return result


class BaseLLMProvider:
    default_model: str = "gpt-4o"

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError
        # Satisfy type checker — never actually reached.
        yield StreamChunk()  # type: ignore[misc]


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-4o",
    ):
        import openai

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key and not base_url:
            logger.warning(
                "OPENAI_API_KEY is not set. Set the environment variable or pass api_key= "
                "to OpenAIProvider. Falling back to 'lm-studio' (for local servers)."
            )
            resolved_key = "lm-studio"

        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.client = openai.AsyncOpenAI(api_key=resolved_key, base_url=resolved_url)
        self.default_model = default_model
        self._supports_stream_usage: bool | None = None  # auto-detected per request

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        formatted_messages = _format_messages_openai(messages)

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": formatted_messages,
            "stream": True,
        }
        # Only request include_usage when auto-detection hasn't failed.
        if self._supports_stream_usage is not False:
            kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = _format_tools_openai(tools)

        start_time = time.monotonic()

        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Retry without stream_options on incompatible local endpoints.
            if "stream_options" in str(e) or "include_usage" in str(e):
                self._supports_stream_usage = False
                kwargs.pop("stream_options", None)
                stream = await self.client.chat.completions.create(**kwargs)
            else:
                raise

        # Track per-tool-call state by index for parallel tool call support.
        tc_by_index: dict[int, dict[str, str]] = {}
        has_tool_call = False
        completion_text_buf = ""
        provider_sent_usage = False

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

            # Usage emitted by provider (e.g. OpenAI): yield & mark so we skip fallback.
            if getattr(chunk, "usage", None):
                u = chunk.usage
                provider_sent_usage = True
                yield StreamChunk(
                    usage=Usage(
                        prompt_tokens=u.prompt_tokens or 0,
                        completion_tokens=u.completion_tokens or 0,
                        total_tokens=u.total_tokens or 0,
                        latency_ms=(time.monotonic() - start_time) * 1000,
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
                    idx = tc_delta.index
                    if idx not in tc_by_index:
                        tc_by_index[idx] = {"id": "", "name": "", "args": ""}
                    entry = tc_by_index[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        entry["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        entry["args"] += tc_delta.function.arguments

            if finish_reason in ("tool_calls", "function_call") or (
                finish_reason == "stop" and has_tool_call
            ):
                for idx, entry in tc_by_index.items():
                    tc_id = entry["id"] or f"call_ollama_{os.urandom(4).hex()}"
                    try:
                        args_dict: dict[str, Any] = (
                            json.loads(entry["args"]) if entry["args"] else {}
                        )
                    except Exception:
                        args_dict = {"raw": entry["args"]}
                    yield StreamChunk(
                        delta_tool_call=ToolCallDelta(
                            id=tc_id, name=entry["name"], arguments=args_dict
                        ),
                        finish_reason="tool_calls",
                    )
                has_tool_call = False
                tc_by_index.clear()
            elif finish_reason:
                yield StreamChunk(finish_reason=finish_reason)

        # Tiktoken fallback — only when provider did NOT emit usage.
        if not provider_sent_usage:
            prompt_tokens = sum(
                _count_tokens(m.get("content", "") or "") for m in formatted_messages
            )
            completion_tokens = _count_tokens(completion_text_buf)
            yield StreamChunk(
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=(time.monotonic() - start_time) * 1000,
                )
            )


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-3-5-sonnet-20241022",
    ):
        import anthropic

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Pass api_key= or set the environment variable."
            )
        self.client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self.default_model = default_model

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt = ""
        formatted_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt += (msg.content or "") + "\n"
            elif msg.role == Role.USER:
                formatted_messages.append({"role": "user", "content": msg.content or ""})
            elif msg.role == Role.ASSISTANT:
                content: list[dict[str, Any]] = []
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
            "max_tokens": 8192,
        }
        if system_prompt:
            kwargs["system"] = system_prompt.strip()
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        start_time = time.monotonic()
        prompt_tokens = 0
        completion_tokens = 0

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        completion_tokens += _count_tokens(text)
                        yield StreamChunk(delta_content=text)
                    elif delta_type == "input_json_delta":
                        # Tool call argument fragment — will be assembled in message_stop
                        pass
                elif event_type == "message_start":
                    msg_obj = getattr(event, "message", None)
                    if msg_obj:
                        usage = getattr(msg_obj, "usage", None)
                        if usage:
                            prompt_tokens = getattr(usage, "input_tokens", 0)
                elif event_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        completion_tokens = getattr(usage, "output_tokens", completion_tokens)
                elif event_type == "content_block_stop":
                    pass

            # Collect tool use blocks from final message
            final_message = await stream.get_final_message()
            for block in final_message.content:
                if getattr(block, "type", None) == "tool_use":
                    yield StreamChunk(
                        delta_tool_call=ToolCallDelta(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        ),
                        finish_reason="tool_calls",
                    )

        yield StreamChunk(
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=(time.monotonic() - start_time) * 1000,
            )
        )


def get_provider(
    provider_name: str = "openai", base_url: str | None = None, **kwargs: Any
) -> BaseLLMProvider:
    if provider_name == "anthropic":
        return AnthropicProvider(**kwargs)
    if provider_name == "ollama":
        url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAIProvider(base_url=url, **kwargs)
    # Defaults to OpenAI or local OpenAI-compatible endpoint
    return OpenAIProvider(base_url=base_url, **kwargs)
