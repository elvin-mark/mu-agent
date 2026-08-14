"""mu-agent: Extensible, multi-provider AI coding agent in Python."""

from .agent import Agent, AgentEvent
from .compaction import ContextCompactor
from .hooks import HookManager
from .llm import AnthropicProvider, BaseLLMProvider, OpenAIProvider, get_provider
from .mcp import MCPManager
from .permissions import PermissionManager, PermissionMode
from .session import SessionManager
from .skills import SkillManager
from .subagent import SubagentManager
from .tools import ToolRegistry, create_default_registry
from .types import Message, Role, StreamChunk, ToolCall, ToolResult, Usage

__all__ = [
    # Core
    "Agent",
    "AgentEvent",
    # LLM Providers
    "AnthropicProvider",
    "BaseLLMProvider",
    "OpenAIProvider",
    "get_provider",
    # Types
    "Message",
    "Role",
    "StreamChunk",
    "ToolCall",
    "ToolResult",
    "Usage",
    # Tools
    "ToolRegistry",
    "create_default_registry",
    # Session
    "SessionManager",
    # Skills
    "SkillManager",
    # MCP
    "MCPManager",
    # Subagents
    "SubagentManager",
    # Hooks
    "HookManager",
    # Permissions
    "PermissionManager",
    "PermissionMode",
    # Compaction
    "ContextCompactor",
]
