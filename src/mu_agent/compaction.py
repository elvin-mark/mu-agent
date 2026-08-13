"""Multi-Tier Context Compaction for Mu Agent."""

from typing import TYPE_CHECKING

from .types import Message, Role

if TYPE_CHECKING:
    from .llm import BaseLLMProvider


class ContextCompactor:
    """Manages multi-tier context compaction strategies."""

    def __init__(self, target_max_tokens: int = 12000):
        self.target_max_tokens = target_max_tokens

    @staticmethod
    def estimate_tokens(messages: list[Message]) -> int:
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

    async def compact(
        self,
        messages: list[Message],
        llm: "BaseLLMProvider | None" = None,
        max_tokens: int | None = None,
    ) -> tuple[list[Message], bool]:
        """Perform multi-tier context compaction.

        Returns:
            (compacted_messages, did_compact)
        """
        budget = max_tokens or self.target_max_tokens
        curr_tokens = self.estimate_tokens(messages)

        # Tier 1 Threshold: 70% of max budget
        tier1_threshold = int(budget * 0.7)
        # Tier 2 Threshold: 90% of max budget
        tier2_threshold = int(budget * 0.9)

        if curr_tokens < tier1_threshold:
            return messages, False

        did_compact = False

        # Tier 1: Truncate large tool outputs in older turns (excluding last 6 turns)
        if len(messages) > 6:
            for msg in messages[:-6]:
                if (
                    msg.role == Role.TOOL
                    and msg.tool_result
                    and len(msg.tool_result.output) > 500
                ):
                    msg.tool_result.output = (
                        msg.tool_result.output[:200]
                        + "\n... [Output truncated by Tier 1 Compaction] ...\n"
                        + msg.tool_result.output[-200:]
                    )
                    did_compact = True

        curr_tokens = self.estimate_tokens(messages)

        # Tier 2: LLM Summarization of older turns if still exceeding 90% capacity
        if curr_tokens >= tier2_threshold and len(messages) > 8 and llm is not None:
            sys_msg = (
                messages[0] if messages and messages[0].role == Role.SYSTEM else None
            )
            start_idx = 1 if sys_msg else 0
            older_turns = messages[start_idx:-6]
            recent_turns = messages[-6:]

            summary_text = await self._summarize_turns(older_turns, llm)
            summary_msg = Message(
                role=Role.USER,
                content=f"[Context Summary of earlier turns]:\n{summary_text}",
            )

            new_messages = []
            if sys_msg:
                new_messages.append(sys_msg)
            new_messages.append(summary_msg)
            new_messages.extend(recent_turns)

            return new_messages, True

        return messages, did_compact

    async def _summarize_turns(
        self, turns: list[Message], llm: "BaseLLMProvider"
    ) -> str:
        """Call LLM to generate a concise summary of older conversation turns."""
        prompt = "Summarize the key findings, user instructions, code edits, and tool results from these turns:\n\n"
        for i, m in enumerate(turns):
            if m.role == Role.USER:
                prompt += f"User: {m.content}\n"
            elif m.role == Role.ASSISTANT:
                prompt += f"Assistant: {m.content or '[Tool Calls]'}\n"
            elif m.role == Role.TOOL and m.tool_result:
                prompt += f"Tool ({m.tool_result.name}): {m.tool_result.output[:300]}\n"

        summary_msgs = [
            Message(
                role=Role.SYSTEM,
                content="You are a context compactor. Provide a concise bulleted summary of key facts and progress.",
            ),
            Message(role=Role.USER, content=prompt),
        ]

        summary = ""
        async for chunk in llm.stream_chat(summary_msgs):
            if chunk.delta_content:
                summary += chunk.delta_content
        return summary.strip() or "Prior turns completed successfully."
