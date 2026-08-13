
import pytest

from mu_agent.agent import Agent
from mu_agent.llm import OpenAIProvider
from mu_agent.subagent import SubagentManager


@pytest.mark.asyncio
async def test_subagent_manager_structure():
    llm = OpenAIProvider(api_key="mock")
    agent = Agent(llm=llm)
    mgr = SubagentManager(parent_agent=agent)
    assert mgr.parent_agent == agent
    assert "spawn_subagent" in agent.tools.handlers
