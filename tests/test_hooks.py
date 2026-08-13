import pytest

from mu_agent.hooks import HookManager, hook


@pytest.mark.asyncio
async def test_hook_registration_and_execution():
    manager = HookManager(plugin_dirs=[])

    pre_called = False

    @hook("pre_tool_call")
    async def sample_pre_hook(name: str, args: dict):
        nonlocal pre_called
        pre_called = True
        args["modified"] = True
        return name, args

    @hook("post_tool_call")
    def sample_post_hook(name: str, output: str) -> str:
        return output + " [sanitized]"

    # Register manually to manager
    manager.register("pre_tool_call", sample_pre_hook)
    manager.register("post_tool_call", sample_post_hook)

    _name, args = await manager.trigger_pre_tool_call("run_command", {"command": "ls"})
    assert pre_called is True
    assert args.get("modified") is True

    out = await manager.trigger_post_tool_call("run_command", "hello world")
    assert out == "hello world [sanitized]"
