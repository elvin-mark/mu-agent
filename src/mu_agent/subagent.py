"""Subagent Delegation & Spawning Subsystem for Mu Agent."""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from .types import Message, Role, ToolCall


class SubagentResult:
    def __init__(self, task_id: str, prompt: str, output: str, is_error: bool = False):
        self.task_id = task_id
        self.prompt = prompt
        self.output = output
        self.is_error = is_error


class SubagentManager:
    def __init__(self, parent_agent):
        self.parent_agent = parent_agent
        self.active_tasks: Dict[str, asyncio.Task] = {}

    async def spawn_subagent(
        self, prompt: str, system_override: Optional[str] = None
    ) -> SubagentResult:
        """Spawn an isolated background subagent task to complete a specific goal."""
        task_id = f"subagent_{str(uuid.uuid4())[:6]}"

        # Import inside function to avoid circular imports
        from .agent import Agent
        from .llm import get_provider

        sub_llm = get_provider(
            self.parent_agent.llm.__class__.__name__.lower().replace("provider", ""),
            default_model=getattr(self.parent_agent.llm, "default_model", "gpt-4o"),
        )

        sub_sys_prompt = system_override or (
            "You are a specialized subagent delegated by Mu Agent to complete a focused task.\n"
            "Be direct, complete the requested objective, and provide a clear final summary."
        )

        sub_agent = Agent(llm=sub_llm, system_prompt=sub_sys_prompt)
        sub_agent.add_user_message(prompt)

        final_output = ""
        try:
            async for event in sub_agent.step():
                if event.type == "content_delta":
                    final_output += event.payload
            return SubagentResult(
                task_id=task_id, prompt=prompt, output=final_output, is_error=False
            )
        except Exception as e:
            return SubagentResult(
                task_id=task_id,
                prompt=prompt,
                output=f"Subagent Error: {str(e)}",
                is_error=True,
            )


def register_subagent_tools(registry, subagent_manager: SubagentManager):
    async def spawn_subagent_handler(args: Dict[str, Any]) -> str:
        prompt = args.get("prompt")
        system_override = args.get("system_prompt")
        if not prompt:
            return "Error: Subagent prompt is required."

        res = await subagent_manager.spawn_subagent(
            prompt=prompt, system_override=system_override
        )
        if res.is_error:
            return f"❌ Subagent ({res.task_id}) failed: {res.output}"
        return f"=== Subagent Task Completed ({res.task_id}) ===\n{res.output}"

    registry.register(
        "spawn_subagent",
        "Spawn an isolated background subagent to solve a sub-task or perform deep research in parallel.",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Specific task prompt instructions for the subagent",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional custom system prompt override for subagent behavior",
                },
            },
            "required": ["prompt"],
        },
        spawn_subagent_handler,
    )
