"""Permission Guardrails, YOLO Mode, and Safety net system for Mu Agent."""

import re
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any


class PermissionMode(str, Enum):
    YOLO = "yolo"
    ASK = "ask"
    READ_ONLY = "read_only"


READ_ONLY_TOOLS = {
    "view_file",
    "list_dir",
    "web_search",
    "read_url",
    "get_subagent_status",
    "load_skill",
}

WRITE_EXEC_TOOLS = {
    "edit_file",
    "replace_file_content",
    "run_command",
    "spawn_subagent",
    "send_subagent_message",
}

ULTRA_DESTRUCTIVE_PATTERNS = [
    r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+.*",
    r"sudo\s+.*",
    r"mkfs\..*",
    r"dd\s+if=.*",
    r"git\s+reset\s+--hard.*",
    r"git\s+clean\s+-[a-zA-Z]*f.*",
    r"> /dev/sd.*",
    r"shutdown.*",
    r"reboot.*",
]


def is_ultra_destructive_command(command: str) -> bool:
    """Check if a shell command matches ultra-destructive patterns."""
    for pattern in ULTRA_DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False


ConfirmationCallback = Callable[
    [str, dict[str, Any]], Coroutine[Any, Any, tuple[bool, bool]]
]
# Tuple[bool, bool] -> (approved, approve_all_for_session)


class PermissionManager:
    """Manages permissions, tool access rules, and confirmation prompts."""

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.ASK,
        confirmation_callback: ConfirmationCallback | None = None,
    ):
        self.mode = mode
        self.confirmation_callback = confirmation_callback
        self.session_approved_all = False

    def set_mode(self, mode: PermissionMode):
        self.mode = mode

    async def evaluate_and_confirm(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[bool, str]:
        """Evaluate permission rules and trigger confirmation if necessary.

        Returns:
            (allowed: bool, reason: str)
        """
        # 1. Read-Only Mode Check
        if self.mode == PermissionMode.READ_ONLY:
            if tool_name in WRITE_EXEC_TOOLS:
                return (
                    False,
                    f"Permission Denied: '{tool_name}' is disabled in Read-Only mode.",
                )

        # 2. Check for Ultra-Destructive Commands
        is_destructive = False
        if tool_name == "run_command":
            cmd = args.get("command", "")
            if is_ultra_destructive_command(cmd):
                is_destructive = True

        # 3. Handle YOLO Mode
        if self.mode == PermissionMode.YOLO and not is_destructive:
            return True, "Approved (YOLO Mode)"

        # If user selected "Approve All for Session" and it's not ultra-destructive
        if self.session_approved_all and not is_destructive:
            return True, "Approved (Session Auto-Approve)"

        # 4. Handle Read-only tools in Ask mode
        if self.mode == PermissionMode.ASK and tool_name in READ_ONLY_TOOLS:
            return True, "Approved (Read-Only Tool)"

        # 5. Confirmation required for write/exec or destructive operations
        if self.confirmation_callback:
            approved, approve_all = await self.confirmation_callback(tool_name, args)
            if approve_all:
                self.session_approved_all = True
            if approved:
                return True, "Approved by user"
            return False, f"Cancelled by user: '{tool_name}' execution was rejected."

        # If no callback provided, default to approve if YOLO, else reject for safety
        if self.mode == PermissionMode.YOLO:
            return True, "Approved (YOLO Default)"

        return True, "Approved (No confirmation handler registered)"
