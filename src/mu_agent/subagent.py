"""Subagent Delegation & Swarm Messaging Subsystem for Mu Agent."""

import asyncio
import uuid
from typing import Any


class SubagentResult:
    def __init__(self, task_id: str, prompt: str, output: str, is_error: bool = False):
        self.task_id = task_id
        self.prompt = prompt
        self.output = output
        self.is_error = is_error


class SubagentInstance:
    """Represents an active or completed subagent worker."""

    def __init__(self, task_id: str, prompt: str, sub_agent: Any):
        self.task_id = task_id
        self.prompt = prompt
        self.sub_agent = sub_agent
        self.status = "running"  # "running", "completed", "error"
        self.inbox: list[str] = []
        self.output_history: list[str] = []
        self.error_message: str | None = None

    def post_message(self, message: str):
        self.inbox.append(message)
        # Add message to subagent message history as user update
        if hasattr(self.sub_agent, "add_user_message"):
            self.sub_agent.add_user_message(
                f"[Inbox Message from Parent/Peer]: {message}"
            )


class SubagentManager:
    """Manages spawning, lifecycle, and inter-subagent swarm messaging."""

    def __init__(self, parent_agent: Any):
        self.parent_agent = parent_agent
        self.subagents: dict[str, SubagentInstance] = {}
        self.active_tasks: dict[str, asyncio.Task] = {}

    async def spawn_subagent(
        self, prompt: str, system_override: str | None = None
    ) -> SubagentResult:
        """Spawn an isolated subagent task to complete a specific goal."""
        task_id = f"subagent_{str(uuid.uuid4())[:6]}"

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

        instance = SubagentInstance(task_id=task_id, prompt=prompt, sub_agent=sub_agent)
        self.subagents[task_id] = instance

        final_output = ""
        try:
            async for event in sub_agent.step():
                if event.type == "content_delta":
                    final_output += event.payload
                    instance.output_history.append(event.payload)

            instance.status = "completed"
            return SubagentResult(
                task_id=task_id, prompt=prompt, output=final_output, is_error=False
            )
        except Exception as e:
            instance.status = "error"
            instance.error_message = str(e)
            return SubagentResult(
                task_id=task_id,
                prompt=prompt,
                output=f"Subagent Error: {e!s}",
                is_error=True,
            )

    def send_message(self, subagent_id: str, message: str) -> str:
        """Send an asynchronous message to a subagent's inbox."""
        if subagent_id not in self.subagents:
            return f"Error: Subagent ID '{subagent_id}' not found."
        instance = self.subagents[subagent_id]
        instance.post_message(message)
        return f"Message delivered to subagent '{subagent_id}' (Status: {instance.status})."

    def get_status(self, subagent_id: str) -> str:
        """Query state, output summary, and inbox messages for a subagent."""
        if subagent_id not in self.subagents:
            return f"Error: Subagent ID '{subagent_id}' not found."
        instance = self.subagents[subagent_id]
        recent_output = "".join(instance.output_history[-5:])
        return (
            f"=== Subagent [{instance.task_id}] Status ===\n"
            f"Status: {instance.status}\n"
            f"Inbox Count: {len(instance.inbox)}\n"
            f"Recent Output:\n{recent_output or '[No output yet]'}"
        )


def register_subagent_tools(registry: Any, subagent_manager: SubagentManager):
    async def spawn_subagent_handler(args: dict[str, Any]) -> str:
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

    async def send_subagent_message_handler(args: dict[str, Any]) -> str:
        subagent_id = args.get("subagent_id")
        message = args.get("message")
        if not subagent_id or not message:
            return "Error: subagent_id and message parameters are required."
        return subagent_manager.send_message(subagent_id, message)

    async def get_subagent_status_handler(args: dict[str, Any]) -> str:
        subagent_id = args.get("subagent_id")
        if not subagent_id:
            return "Error: subagent_id parameter is required."
        return subagent_manager.get_status(subagent_id)

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

    registry.register(
        "send_subagent_message",
        "Send a message into a running subagent's inbox queue for multi-agent swarm coordination.",
        {
            "type": "object",
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "Target subagent ID (e.g. subagent_a1b2c3)",
                },
                "message": {
                    "type": "string",
                    "description": "Message content to deliver to the subagent",
                },
            },
            "required": ["subagent_id", "message"],
        },
        send_subagent_message_handler,
    )

    registry.register(
        "get_subagent_status",
        "Query status, progress output, and inbox state of a subagent.",
        {
            "type": "object",
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "Subagent ID to query",
                },
            },
            "required": ["subagent_id"],
        },
        get_subagent_status_handler,
    )
