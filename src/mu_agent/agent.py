"""Agent Core Runtime orchestrating tool calls, message context, and execution loop."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from .compaction import ContextCompactor
from .hooks import HookManager
from .llm import BaseLLMProvider
from .mcp import MCPManager
from .permissions import PermissionManager
from .session import SessionManager, load_project_instructions
from .skills import SkillManager
from .subagent import SubagentManager, register_subagent_tools
from .tools import ToolRegistry, create_default_registry, register_skill_tools
from .types import Message, Role, ToolCall, ToolCallDelta, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are Mu, an expert AI coding assistant.
You have access to tools for interacting with the filesystem, running shell commands, and searching the web.
Always think carefully, break down complex tasks, verify code edits, and keep your responses clear and helpful.
"""


class AgentEvent:
    __slots__ = ("type", "payload")

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
        permission_manager: PermissionManager | None = None,
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
        self.permission_manager = permission_manager or PermissionManager()

        # Register permission hook
        async def _perm_hook(tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            allowed, reason = await self.permission_manager.evaluate_and_confirm(
                tool_name, args
            )
            if not allowed:
                raise PermissionError(reason)
            return tool_name, args

        self.hook_manager.register("pre_tool_call", _perm_hook)

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
        # Clone the registry to avoid cross-instance side effects
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

    def add_user_message(self, text: str) -> None:
        msg = Message(role=Role.USER, content=text)
        self.messages.append(msg)
        self.session_manager.save_message(msg)

    async def compact_context(self) -> None:
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
            pending_tool_calls: list[ToolCallDelta] = []

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
                    pending_tool_calls.append(chunk.delta_tool_call)
                if chunk.usage:
                    yield AgentEvent("usage", chunk.usage)

            if pending_tool_calls:
                # Build assistant message with all tool calls for this turn.
                tool_calls = [
                    ToolCall(
                        id=tc.id or f"call_{i}",
                        name=tc.name,
                        arguments=tc.arguments if isinstance(tc.arguments, dict) else {},
                    )
                    for i, tc in enumerate(pending_tool_calls)
                ]
                asst_msg = Message(
                    role=Role.ASSISTANT,
                    content=assistant_content or None,
                    tool_calls=tool_calls,
                )
                self.messages.append(asst_msg)
                self.session_manager.save_message(asst_msg)

                # Execute all tool calls for this turn (sequential; parallel is a future opt).
                for tc in tool_calls:
                    tool_name = tc.name
                    tool_output = ""
                    is_error = False

                    try:
                        (
                            tool_name,
                            tool_args,
                        ) = await self.hook_manager.trigger_pre_tool_call(
                            tc.name, tc.arguments
                        )
                        yield AgentEvent("tool_call_start", tc)
                        raw_output = await self.tools.execute(tool_name, tool_args)
                        tool_output = await self.hook_manager.trigger_post_tool_call(
                            tool_name, raw_output
                        )
                    except PermissionError as pe:
                        tool_output = str(pe)
                        is_error = True
                        logger.info("Permission denied for tool '%s': %s", tc.name, pe)
                    except Exception as exc:
                        tool_output = f"Tool '{tc.name}' raised an error: {exc!s}"
                        is_error = True
                        logger.warning("Tool '%s' execution failed: %s", tc.name, exc)

                    yield AgentEvent(
                        "tool_call_end", {"call": tc, "output": tool_output, "is_error": is_error}
                    )

                    tool_res = ToolResult(
                        tool_call_id=tc.id,
                        name=tool_name,
                        output=tool_output,
                        is_error=is_error,
                    )
                    tool_msg = Message(
                        role=Role.TOOL,
                        tool_result=tool_res,
                    )
                    self.messages.append(tool_msg)
                    self.session_manager.save_message(tool_msg)

            else:
                # No tool calls — agent has produced its final response for this turn.
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
