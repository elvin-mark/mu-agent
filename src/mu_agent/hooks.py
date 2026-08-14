"""Lifecycle Hook & Plugin System for Mu Agent."""

from __future__ import annotations

import importlib.util
import inspect
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Type for hook functions
HookFunc = Callable[..., Coroutine[Any, Any, Any] | Any]

_KNOWN_EVENTS = frozenset(
    {"pre_tool_call", "post_tool_call", "on_turn_start", "on_turn_end", "on_context_compact"}
)


def hook(event_name: str) -> Callable[[HookFunc], HookFunc]:
    """Decorator to register a function as a lifecycle hook.

    Note: This decorator registers into a *per-manager* registry at call time
    by storing the function and event name. HookManager.load_plugins() will
    pick up all @hook-decorated functions from loaded plugin modules.

    Example::

        @hook("pre_tool_call")
        async def my_pre_tool_hook(tool_name: str, args: dict) -> tuple[str, dict]:
            ...
    """

    def decorator(fn: HookFunc) -> HookFunc:
        # Tag the function so HookManager.load_plugins can discover it.
        if not hasattr(fn, "_mu_hook_events"):
            fn._mu_hook_events = []  # type: ignore[attr-defined]
        fn._mu_hook_events.append(event_name)  # type: ignore[attr-defined]
        return fn

    return decorator


class HookManager:
    """Manages discovery and invocation of agent lifecycle hooks.

    Each HookManager instance has its own isolated registry — no global mutable state.
    """

    def __init__(self, plugin_dirs: list[Path] | None = None):
        self.plugin_dirs = plugin_dirs or [
            Path(".mu/plugins"),
            Path.home() / ".mu" / "plugins",
        ]
        # Instance-scoped registry — no shared global state.
        self.registry: dict[str, list[HookFunc]] = {event: [] for event in _KNOWN_EVENTS}

    def register(self, event_name: str, fn: HookFunc) -> None:
        if event_name not in self.registry:
            self.registry[event_name] = []
        if fn not in self.registry[event_name]:
            self.registry[event_name].append(fn)

    def load_plugins(self) -> None:
        """Discover and dynamically load Python scripts from plugin directories."""
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists() or not plugin_dir.is_dir():
                continue
            for file_path in sorted(plugin_dir.glob("*.py")):
                if file_path.name.startswith("_"):
                    continue
                module_name = f"mu_plugin_{file_path.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        # Auto-register all @hook-decorated functions from the module.
                        for attr in vars(mod).values():
                            events = getattr(attr, "_mu_hook_events", None)
                            if events:
                                for event_name in events:
                                    self.register(event_name, attr)
                        logger.debug("Loaded plugin: %s", file_path)
                except Exception as e:
                    # Log to logger instead of print — does not corrupt TUI output.
                    logger.warning("Failed to load plugin %s: %s", file_path, e)

    async def trigger_pre_tool_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Invoke pre_tool_call hooks sequentially. Hooks may alter args or raise exceptions."""
        curr_name, curr_args = tool_name, dict(args)
        for fn in self.registry.get("pre_tool_call", []):
            try:
                res = (
                    await fn(curr_name, curr_args)
                    if inspect.iscoroutinefunction(fn)
                    else fn(curr_name, curr_args)
                )
                if isinstance(res, tuple) and len(res) == 2:
                    curr_name, curr_args = res[0], res[1]
                elif isinstance(res, dict):
                    curr_args = res
            except PermissionError:
                raise  # Propagate permission errors verbatim
            except Exception as e:
                logger.warning("pre_tool_call hook %s raised: %s", fn.__name__, e)
        return curr_name, curr_args

    async def trigger_post_tool_call(self, tool_name: str, output: str) -> str:
        """Invoke post_tool_call hooks sequentially to inspect or sanitize tool output."""
        curr_output = output
        for fn in self.registry.get("post_tool_call", []):
            try:
                res = (
                    await fn(tool_name, curr_output)
                    if inspect.iscoroutinefunction(fn)
                    else fn(tool_name, curr_output)
                )
                if isinstance(res, str):
                    curr_output = res
            except Exception as e:
                logger.warning("post_tool_call hook %s raised: %s", fn.__name__, e)
        return curr_output

    async def trigger_event(self, event_name: str, payload: Any = None) -> None:
        """Trigger generic lifecycle events (e.g. on_turn_start, on_turn_end)."""
        for fn in self.registry.get(event_name, []):
            try:
                if inspect.iscoroutinefunction(fn):
                    await fn(payload)
                else:
                    fn(payload)
            except Exception as e:
                logger.warning("Hook %s for event '%s' raised: %s", fn.__name__, event_name, e)
