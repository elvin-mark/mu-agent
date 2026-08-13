import pytest

from mu_agent.compaction import ContextCompactor
from mu_agent.types import Message, Role, ToolResult


@pytest.mark.asyncio
async def test_context_compactor_tier1():
    compactor = ContextCompactor(target_max_tokens=100)

    long_output = "X" * 1000
    messages = [
        Message(role=Role.SYSTEM, content="System prompt"),
        Message(role=Role.USER, content="User prompt 1"),
        Message(role=Role.ASSISTANT, content="Assistant 1"),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(
                tool_call_id="1", name="list_dir", output=long_output
            ),
        ),
        Message(role=Role.USER, content="User 2"),
        Message(role=Role.ASSISTANT, content="Assistant 2"),
        Message(role=Role.USER, content="User 3"),
        Message(role=Role.ASSISTANT, content="Assistant 3"),
        Message(role=Role.USER, content="User 4"),
        Message(role=Role.ASSISTANT, content="Assistant 4"),
    ]

    compacted, did_compact = await compactor.compact(messages, max_tokens=100)
    assert did_compact is True
    assert "Output truncated by Tier 1 Compaction" in compacted[3].tool_result.output
