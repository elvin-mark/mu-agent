"""Session management, persistence (.pi/sessions), and project instructions (AGENTS.md)."""

import json
import os
import time
import uuid
from typing import Any

from .types import Message

MU_DIR = ".mu" if os.path.exists(".mu") else ".pi"
SESSIONS_DIR = os.path.join(MU_DIR, "sessions")


class SessionManager:
    def __init__(self, session_id: str | None = None, sessions_dir: str | None = None):

        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.sessions_dir = sessions_dir or SESSIONS_DIR
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.session_file = os.path.join(
            self.sessions_dir, f"session_{self.session_id}.jsonl"
        )

    def save_message(self, message: Message):
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(message.model_dump_json() + "\n")

    def load_session(self) -> list[Message]:
        if not os.path.exists(self.session_file):
            return []
        messages = []
        with open(self.session_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    messages.append(Message.model_validate(data))
        return messages

    @staticmethod
    def list_sessions() -> list[dict[str, Any]]:
        if not os.path.exists(SESSIONS_DIR):
            return []
        sessions = []
        for file in os.listdir(SESSIONS_DIR):
            if file.startswith("session_") and file.endswith(".jsonl"):
                path = os.path.join(SESSIONS_DIR, file)
                sid = file.replace("session_", "").replace(".jsonl", "")
                mtime = os.path.getmtime(path)
                sessions.append(
                    {"session_id": sid, "path": path, "mtime": time.ctime(mtime)}
                )
        return sorted(sessions, key=lambda x: x["mtime"], reverse=True)


def load_project_instructions(root_dir: str = ".") -> str:
    """Load project custom instructions from AGENTS.md or .pi/SYSTEM.md if available."""
    instructions = []

    agents_md = os.path.join(root_dir, "AGENTS.md")
    if os.path.exists(agents_md):
        try:
            with open(agents_md, encoding="utf-8") as f:
                instructions.append(f"--- Instructions from AGENTS.md ---\n{f.read()}")
        except Exception:
            pass

    mu_system = os.path.join(root_dir, MU_DIR, "SYSTEM.md")
    if os.path.exists(mu_system):
        try:
            with open(mu_system, encoding="utf-8") as f:
                instructions.append(
                    f"--- Instructions from {MU_DIR}/SYSTEM.md ---\n{f.read()}"
                )
        except Exception:
            pass

    return "\n\n".join(instructions)
