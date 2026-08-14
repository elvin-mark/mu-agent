"""Permission Guardrails, YOLO Mode, and Safety net system for Mu Agent."""

import enum
import re
from collections.abc import Callable, Coroutine
from typing import Any


class PermissionMode(enum.StrEnum):
    YOLO = "yolo"
    ASK = "ask"
    READ_ONLY = "read_only"


READ_ONLY_TOOLS = frozenset(
    {
        "view_file",
        "list_dir",
        "web_search",
        "read_url",
        "get_subagent_status",
        "load_skill",
    }
)

WRITE_EXEC_TOOLS = frozenset(
    {
        "edit_file",
        "replace_file_content",
        "apply_patch",
        "run_command",
        "spawn_subagent",
        "send_subagent_message",
    }
)

ULTRA_DESTRUCTIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+.*", re.IGNORECASE),
    re.compile(r"sudo\s+.*", re.IGNORECASE),
    re.compile(r"mkfs\..*", re.IGNORECASE),
    re.compile(r"dd\s+if=.*", re.IGNORECASE),
    re.compile(r"git\s+reset\s+--hard.*", re.IGNORECASE),
    re.compile(r"git\s+clean\s+-[a-zA-Z]*f.*", re.IGNORECASE),
    re.compile(r">\s*/dev/sd.*", re.IGNORECASE),
    re.compile(r"shutdown.*", re.IGNORECASE),
    re.compile(r"reboot.*", re.IGNORECASE),
]


def is_ultra_destructive_command(command: str) -> bool:
    """Check if a shell command matches ultra-destructive patterns."""
    return any(p.search(command) for p in ULTRA_DESTRUCTIVE_PATTERNS)


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

    def set_mode(self, mode: PermissionMode) -> None:
        self.mode = mode

    async def evaluate_and_confirm(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[bool, str]:
        """Evaluate permission rules and trigger confirmation if necessary.

        Returns:
            (allowed: bool, reason: str)
        """
        # 1. Read-Only Mode: allow ONLY explicitly whitelisted read tools.
        if self.mode == PermissionMode.READ_ONLY:
            if tool_name in READ_ONLY_TOOLS:
                return True, "Approved (Read-Only Tool)"
            return (
                False,
                f"Permission Denied: '{tool_name}' is not permitted in Read-Only mode.",
            )

        # 2. Check for Ultra-Destructive Commands (always require confirmation)
        is_destructive = False
        if tool_name == "run_command":
            cmd = args.get("command", "")
            if is_ultra_destructive_command(cmd):
                is_destructive = True

        # 3. YOLO Mode: approve everything except ultra-destructive commands
        if self.mode == PermissionMode.YOLO and not is_destructive:
            return True, "Approved (YOLO Mode)"

        # Session-level auto-approve (set when user chose "approve all")
        if self.session_approved_all and not is_destructive:
            return True, "Approved (Session Auto-Approve)"

        # 4. In ASK mode, read-only tools are always auto-approved
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

        # 6. No callback registered — fail safe: deny by default in non-YOLO modes
        if self.mode == PermissionMode.YOLO:
            return True, "Approved (YOLO Default)"

        return (
            False,
            f"Permission Denied: no confirmation handler registered for '{tool_name}'. "
            "Set --permission=yolo to auto-approve, or provide a confirmation_callback.",
        )
