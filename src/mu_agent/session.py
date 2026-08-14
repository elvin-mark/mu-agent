"""Session management, persistence (.mu/sessions), and project instructions (AGENTS.md)."""

import json
import logging
import os
import uuid
from typing import Any

from .types import Message

logger = logging.getLogger(__name__)


def _resolve_mu_dir(root: str = ".") -> str:
    """Resolve config dir: prefer .mu, fall back to .pi for backward compatibility."""
    mu = os.path.join(root, ".mu")
    pi = os.path.join(root, ".pi")
    if os.path.exists(mu):
        return ".mu"
    if os.path.exists(pi):
        return ".pi"
    return ".mu"  # default for new projects


class SessionManager:
    def __init__(
        self,
        session_id: str | None = None,
        sessions_dir: str | None = None,
        root_dir: str = ".",
    ):
        self.root_dir = root_dir
        self.mu_dir = _resolve_mu_dir(root_dir)
        default_sessions_dir = os.path.join(root_dir, self.mu_dir, "sessions")
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.sessions_dir = sessions_dir or default_sessions_dir
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.session_file = os.path.join(
            self.sessions_dir, f"session_{self.session_id}.jsonl"
        )

    def save_message(self, message: Message) -> None:
        try:
            with open(self.session_file, "a", encoding="utf-8") as f:
                f.write(message.model_dump_json() + "\n")
        except OSError as e:
            logger.warning("Failed to persist message to session: %s", e)

    def load_session(self) -> list[Message]:
        if not os.path.exists(self.session_file):
            return []
        messages: list[Message] = []
        with open(self.session_file, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    messages.append(Message.model_validate(data))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "Skipping corrupt session line %d in %s: %s",
                        lineno,
                        self.session_file,
                        e,
                    )
        return messages

    def list_sessions(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.sessions_dir):
            return []
        sessions: list[dict[str, Any]] = []
        for file in os.listdir(self.sessions_dir):
            if file.startswith("session_") and file.endswith(".jsonl"):
                path = os.path.join(self.sessions_dir, file)
                sid = file.removeprefix("session_").removesuffix(".jsonl")
                mtime = os.path.getmtime(path)  # numeric float — sort correctly
                size = os.path.getsize(path)
                sessions.append(
                    {"session_id": sid, "path": path, "mtime": mtime, "size_bytes": size}
                )
        # Sort by numeric mtime descending (most recent first) — not ctime string!
        return sorted(sessions, key=lambda x: x["mtime"], reverse=True)


def load_project_instructions(root_dir: str = ".") -> str:
    """Load project custom instructions from AGENTS.md or .mu/SYSTEM.md if available."""
    mu_dir = _resolve_mu_dir(root_dir)
    instructions: list[str] = []

    for candidates in [
        (os.path.join(root_dir, "AGENTS.md"), "AGENTS.md"),
        (os.path.join(root_dir, mu_dir, "SYSTEM.md"), f"{mu_dir}/SYSTEM.md"),
    ]:
        path, label = candidates
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    instructions.append(f"--- Instructions from {label} ---\n{f.read()}")
            except OSError as e:
                logger.warning("Could not read project instructions from %s: %s", path, e)

    return "\n\n".join(instructions)
