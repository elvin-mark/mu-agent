"""Lifecycle Hook & Plugin System for Mu Agent."""

import importlib.util
import inspect
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

# Type for hook functions
HookFunc = Callable[..., Coroutine[Any, Any, Any] | Any]

_GLOBAL_HOOK_REGISTRY: dict[str, list[HookFunc]] = {
    "pre_tool_call": [],
    "post_tool_call": [],
    "on_turn_start": [],
    "on_turn_end": [],
    "on_context_compact": [],
}


def hook(event_name: str):
    """Decorator to register a function as a lifecycle hook.

    Example:
        @hook("pre_tool_call")
        async def my_pre_tool_hook(tool_name: str, args: dict):
            ...
    """

    def decorator(fn: HookFunc) -> HookFunc:
        if event_name not in _GLOBAL_HOOK_REGISTRY:
            _GLOBAL_HOOK_REGISTRY[event_name] = []
        if fn not in _GLOBAL_HOOK_REGISTRY[event_name]:
            _GLOBAL_HOOK_REGISTRY[event_name].append(fn)
        return fn

    return decorator


class HookManager:
    """Manages discovery and invocation of agent lifecycle hooks."""

    def __init__(self, plugin_dirs: list[Path] | None = None):
        self.plugin_dirs = plugin_dirs or [
            Path(".mu/plugins"),
            Path.home() / ".mu" / "plugins",
        ]
        self.registry = _GLOBAL_HOOK_REGISTRY

    def load_plugins(self):
        """Discover and dynamically load Python scripts from plugin directories."""
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists() or not plugin_dir.is_dir():
                continue
            for file_path in plugin_dir.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                module_name = f"mu_plugin_{file_path.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, file_path
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                except Exception as e:
                    print(f"Warning: Failed to load plugin {file_path}: {e}")

    def register(self, event_name: str, fn: HookFunc):
        if event_name not in self.registry:
            self.registry[event_name] = []
        if fn not in self.registry[event_name]:
            self.registry[event_name].append(fn)

    async def trigger_pre_tool_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Invoke pre_tool_call hooks sequentially. Hooks may alter args or raise exceptions."""
        curr_name, curr_args = tool_name, dict(args)
        for fn in self.registry.get("pre_tool_call", []):
            res = (
                await fn(curr_name, curr_args)
                if inspect.iscoroutinefunction(fn)
                else fn(curr_name, curr_args)
            )
            if isinstance(res, tuple) and len(res) == 2:
                curr_name, curr_args = res[0], res[1]
            elif isinstance(res, dict):
                curr_args = res
        return curr_name, curr_args

    async def trigger_post_tool_call(self, tool_name: str, output: str) -> str:
        """Invoke post_tool_call hooks sequentially to inspect or sanitize tool output."""
        curr_output = output
        for fn in self.registry.get("post_tool_call", []):
            res = (
                await fn(tool_name, curr_output)
                if inspect.iscoroutinefunction(fn)
                else fn(tool_name, curr_output)
            )
            if isinstance(res, str):
                curr_output = res
        return curr_output

    async def trigger_event(self, event_name: str, payload: Any = None):
        """Trigger generic lifecycle events (e.g. on_turn_start, on_turn_end)."""
        for fn in self.registry.get(event_name, []):
            if inspect.iscoroutinefunction(fn):
                await fn(payload)
            else:
                fn(payload)
