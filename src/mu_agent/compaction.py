"""Multi-Tier Context Compaction for Mu Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .types import Message, Role

if TYPE_CHECKING:
    from .llm import BaseLLMProvider

logger = logging.getLogger(__name__)


def _estimate_tokens(messages: list[Message]) -> int:
    """Estimate token count across conversation messages (rough heuristic: 4 chars ~ 1 token)."""
    total_chars = 0
    for msg in messages:
        if msg.content:
            total_chars += len(msg.content)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total_chars += len(tc.name) + len(str(tc.arguments))
        if msg.tool_result:
            total_chars += len(msg.tool_result.output)
    return total_chars // 4


def _find_safe_slice_index(messages: list[Message], from_end: int = 6) -> int:
    """Find a safe slice index that doesn't split assistant+tool turn pairs.

    An assistant message with tool_calls MUST be immediately followed by its tool
    result message, or the API will reject the conversation with 400 Bad Request.
    This function scans backward from (len - from_end) to find the nearest safe boundary.
    """
    target = max(1, len(messages) - from_end)
    # Walk backward until we're not in the middle of a tool-call turn.
    idx = target
    while idx > 1:
        msg = messages[idx - 1]
        # If the previous message is an assistant with tool_calls, we can't slice here —
        # the next message (messages[idx]) is its tool result and must stay paired.
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            idx -= 1
        else:
            break
    return idx


class ContextCompactor:
    """Manages multi-tier context compaction strategies."""

    def __init__(self, target_max_tokens: int = 12000):
        self.target_max_tokens = target_max_tokens

    async def compact(
        self,
        messages: list[Message],
        llm: BaseLLMProvider | None = None,
        max_tokens: int | None = None,
    ) -> tuple[list[Message], bool]:
        """Perform multi-tier context compaction.

        Returns:
            (compacted_messages, did_compact)
        """
        budget = max_tokens or self.target_max_tokens
        curr_tokens = _estimate_tokens(messages)

        tier1_threshold = int(budget * 0.7)
        tier2_threshold = int(budget * 0.9)

        if curr_tokens < tier1_threshold:
            return messages, False

        did_compact = False
        safe_idx = _find_safe_slice_index(messages, from_end=6)

        # Tier 1: Truncate large tool outputs in older turns — never mutate in-place.
        if safe_idx > 1:
            new_messages: list[Message] = []
            for i, msg in enumerate(messages):
                if (
                    i < safe_idx
                    and msg.role == Role.TOOL
                    and msg.tool_result
                    and len(msg.tool_result.output) > 500
                ):
                    # Create a new ToolResult (immutable replacement) instead of mutation.
                    truncated_result = msg.tool_result.model_copy(
                        update={
                            "output": (
                                msg.tool_result.output[:200]
                                + "\n... [Output truncated by Tier 1 Compaction] ...\n"
                                + msg.tool_result.output[-200:]
                            )
                        }
                    )
                    new_messages.append(msg.model_copy(update={"tool_result": truncated_result}))
                    did_compact = True
                else:
                    new_messages.append(msg)
            messages = new_messages

        curr_tokens = _estimate_tokens(messages)

        # Tier 2: LLM Summarization if still exceeding 90% capacity.
        if curr_tokens >= tier2_threshold and len(messages) > 8 and llm is not None:
            sys_msg = messages[0] if messages and messages[0].role == Role.SYSTEM else None
            start_idx = 1 if sys_msg else 0

            safe_idx = _find_safe_slice_index(messages, from_end=6)
            older_turns = messages[start_idx:safe_idx]
            recent_turns = messages[safe_idx:]

            if older_turns:
                summary_text = await self._summarize_turns(older_turns, llm)
                summary_msg = Message(
                    role=Role.USER,
                    content=f"[Context Summary of earlier turns]:\n{summary_text}",
                )
                compacted: list[Message] = []
                if sys_msg:
                    compacted.append(sys_msg)
                compacted.append(summary_msg)
                compacted.extend(recent_turns)
                logger.debug(
                    "Tier 2 compaction: %d → %d messages", len(messages), len(compacted)
                )
                return compacted, True

        return messages, did_compact

    async def _summarize_turns(
        self, turns: list[Message], llm: BaseLLMProvider
    ) -> str:
        """Call LLM to generate a concise summary of older conversation turns."""
        prompt_parts: list[str] = [
            "Summarize the key findings, user instructions, code edits, and tool results from these turns:\n"
        ]
        for m in turns:
            if m.role == Role.USER and m.content:
                prompt_parts.append(f"User: {m.content}")
            elif m.role == Role.ASSISTANT:
                prompt_parts.append(f"Assistant: {m.content or '[Tool Calls]'}")
            elif m.role == Role.TOOL and m.tool_result:
                prompt_parts.append(
                    f"Tool ({m.tool_result.name}): {m.tool_result.output[:300]}"
                )

        summary_msgs = [
            Message(
                role=Role.SYSTEM,
                content="You are a context compactor. Provide a concise bulleted summary of key facts and progress.",
            ),
            Message(role=Role.USER, content="\n".join(prompt_parts)),
        ]

        summary = ""
        async for chunk in llm.stream_chat(summary_msgs):
            if chunk.delta_content:
                summary += chunk.delta_content
        return summary.strip() or "Prior turns completed successfully."
