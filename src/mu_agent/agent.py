"""Agent Core Runtime orchestrating tool calls, message context, and execution loop."""

from collections.abc import AsyncIterator
from typing import Any

from .compaction import ContextCompactor
from .hooks import HookManager
from .llm import BaseLLMProvider
from .mcp import MCPManager
from .session import SessionManager, load_project_instructions
from .skills import SkillManager
from .subagent import SubagentManager, register_subagent_tools
from .tools import ToolRegistry, create_default_registry, register_skill_tools
from .types import Message, Role, ToolCall, ToolResult

DEFAULT_SYSTEM_PROMPT = """You are Mu, an expert AI coding assistant.
You have access to tools for interacting with the filesystem, running shell commands, and searching the web.
Always think carefully, break down complex tasks, verify code edits, and keep your responses clear and helpful.
"""


class AgentEvent:
    def __init__(self, type_: str, payload: Any = None):
        self.type = type_
        self.payload = payload


class Agent:
    def __init__(
        self,
        llm: BaseLLMProvider,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tool_registry: ToolRegistry | None = None,
        skill_manager: SkillManager | None = None,
        mcp_manager: MCPManager | None = None,
        hook_manager: HookManager | None = None,
        compactor: ContextCompactor | None = None,
        max_turns: int = 25,
        session_manager: SessionManager | None = None,
        max_history_tokens_estimate: int = 12000,
    ):
        self.llm = llm
        self.skill_manager = skill_manager or SkillManager()
        self.mcp_manager = mcp_manager or MCPManager()
        self.hook_manager = hook_manager or HookManager()
        self.hook_manager.load_plugins()

        self.compactor = compactor or ContextCompactor(
            target_max_tokens=max_history_tokens_estimate
        )

        self.subagent_manager = SubagentManager(parent_agent=self)

        # Load AGENTS.md / .mu/SYSTEM.md project instructions
        proj_instructions = load_project_instructions()
        if proj_instructions:
            system_prompt = (
                f"{system_prompt}\n\nProject Instructions:\n{proj_instructions}"
            )

        # Append Skill metadata prompt
        skills_summary = self.skill_manager.get_skill_summary_prompt()
        if skills_summary:
            system_prompt = f"{system_prompt}\n\n{skills_summary}"

        self.system_prompt = system_prompt
        self.tools = tool_registry or create_default_registry()
        register_skill_tools(self.tools, self.skill_manager)
        register_subagent_tools(self.tools, self.subagent_manager)

        self.max_turns = max_turns
        self.session_manager = session_manager or SessionManager()
        self.max_history_tokens_estimate = max_history_tokens_estimate

        # Load existing session or init system message
        existing_msgs = self.session_manager.load_session()
        if existing_msgs:
            self.messages = existing_msgs
        else:
            sys_msg = Message(role=Role.SYSTEM, content=system_prompt)
            self.messages = [sys_msg]
            self.session_manager.save_message(sys_msg)

    def add_user_message(self, text: str):
        msg = Message(role=Role.USER, content=text)
        self.messages.append(msg)
        self.session_manager.save_message(msg)

    async def compact_context(self):
        """Auto-compact context if history exceeds max estimated token threshold."""
        self.messages, did_compact = await self.compactor.compact(
            messages=self.messages,
            llm=self.llm,
            max_tokens=self.max_history_tokens_estimate,
        )
        if did_compact:
            await self.hook_manager.trigger_event("on_context_compact", self.messages)

    async def step(self) -> AsyncIterator[AgentEvent]:
        turns = 0
        while turns < self.max_turns:
            turns += 1
            assistant_content = ""
            pending_tool_call: dict[str, Any] | None = None

            await self.compact_context()
            await self.hook_manager.trigger_event("on_turn_start", turns)
            yield AgentEvent("step_start", {"turn": turns})

            async for chunk in self.llm.stream_chat(
                messages=self.messages,
                tools=self.tools.schemas if self.tools else None,
            ):
                if chunk.delta_content:
                    assistant_content += chunk.delta_content
                    yield AgentEvent("content_delta", chunk.delta_content)
                if chunk.delta_tool_call:
                    pending_tool_call = chunk.delta_tool_call
                if chunk.usage:
                    yield AgentEvent("usage", chunk.usage)

            if pending_tool_call:
                tc = ToolCall(
                    id=pending_tool_call["id"],
                    name=pending_tool_call["name"],
                    arguments=pending_tool_call["arguments"],
                )
                asst_msg = Message(
                    role=Role.ASSISTANT,
                    content=assistant_content or None,
                    tool_calls=[tc],
                )
                self.messages.append(asst_msg)
                self.session_manager.save_message(asst_msg)

                # Pre-tool lifecycle hook
                tool_name, tool_args = await self.hook_manager.trigger_pre_tool_call(
                    tc.name, tc.arguments
                )

                yield AgentEvent("tool_call_start", tc)
                raw_output = await self.tools.execute(tool_name, tool_args)

                # Post-tool lifecycle hook
                tool_output = await self.hook_manager.trigger_post_tool_call(
                    tool_name, raw_output
                )
                yield AgentEvent("tool_call_end", {"call": tc, "output": tool_output})

                tool_res = ToolResult(
                    tool_call_id=tc.id,
                    name=tool_name,
                    output=tool_output,
                )
                tool_msg = Message(
                    role=Role.TOOL,
                    tool_result=tool_res,
                )
                self.messages.append(tool_msg)
                self.session_manager.save_message(tool_msg)
            else:
                if assistant_content:
                    asst_msg = Message(
                        role=Role.ASSISTANT,
                        content=assistant_content,
                    )
                    self.messages.append(asst_msg)
                    self.session_manager.save_message(asst_msg)
                await self.hook_manager.trigger_event("on_turn_end", turns)
                yield AgentEvent("step_complete")
                break
