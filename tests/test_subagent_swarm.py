import pytest

from mu_agent.agent import Agent
from mu_agent.llm import OpenAIProvider
from mu_agent.subagent import SubagentManager


@pytest.mark.asyncio
async def test_subagent_swarm_messaging():
    llm = OpenAIProvider(api_key="mock")
    agent = Agent(llm=llm)
    mgr = SubagentManager(parent_agent=agent)

    # Check tools registration
    assert "send_subagent_message" in agent.tools.handlers
    assert "get_subagent_status" in agent.tools.handlers

    # Test status of non-existent subagent
    status_err = mgr.get_status("invalid_id")
    assert "not found" in status_err

    # Test sending message to non-existent subagent
    msg_err = mgr.send_message("invalid_id", "Hello!")
    assert "not found" in msg_err
