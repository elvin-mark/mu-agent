"""pi-agent-py: Unified AI agent in Python inspired by Pi."""

import enum
from typing import Any

from pydantic import BaseModel


class Role(enum.StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    output: str
    is_error: bool = False


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class StreamChunk(BaseModel):
    delta_content: str | None = None
    delta_tool_call: dict[str, Any] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
