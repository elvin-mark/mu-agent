"""Core type definitions and Pydantic models for mu-agent."""

import enum
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


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

    @model_validator(mode="after")
    def validate_consistency(self) -> "Message":
        if self.role == Role.TOOL and self.tool_result is None:
            raise ValueError("Messages with role='tool' must include tool_result.")
        if self.tool_calls and self.role != Role.ASSISTANT:
            raise ValueError("Only assistant messages can have tool_calls.")
        return self

    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)

    def is_tool_result(self) -> bool:
        return self.role == Role.TOOL and self.tool_result is not None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    @field_validator("total_tokens", mode="before")
    @classmethod
    def default_total(cls, v: int, info: Any) -> int:
        if v == 0:
            data = info.data
            return data.get("prompt_tokens", 0) + data.get("completion_tokens", 0)
        return v


class ToolCallDelta(BaseModel):
    """Structured type for in-progress streaming tool call fragments."""

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] | str = ""


class StreamChunk(BaseModel):
    delta_content: str | None = None
    delta_tool_call: ToolCallDelta | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
