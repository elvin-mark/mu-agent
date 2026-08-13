import pytest

from mu_agent.permissions import (
    PermissionManager,
    PermissionMode,
    is_ultra_destructive_command,
)


@pytest.mark.asyncio
async def test_permission_modes():
    # 1. Read-Only Mode
    mgr_ro = PermissionManager(mode=PermissionMode.READ_ONLY)
    allowed, reason = await mgr_ro.evaluate_and_confirm(
        "edit_file", {"path": "test.txt", "content": "hi"}
    )
    assert allowed is False
    assert "Read-Only mode" in reason

    allowed_ro, _ = await mgr_ro.evaluate_and_confirm("view_file", {"path": "test.txt"})
    assert allowed_ro is True

    # 2. YOLO Mode (Normal write allowed)
    mgr_yolo = PermissionManager(mode=PermissionMode.YOLO)
    allowed_yolo, _ = await mgr_yolo.evaluate_and_confirm(
        "run_command", {"command": "ls -l"}
    )
    assert allowed_yolo is True

    # 3. Ultra-destructive command detection
    assert is_ultra_destructive_command("rm -rf /") is True
    assert is_ultra_destructive_command("sudo rm -f app.db") is True
    assert is_ultra_destructive_command("git reset --hard") is True
    assert is_ultra_destructive_command("ls -la") is False

    # 4. Confirmation Callback in Ask Mode
    confirmed = False

    async def mock_callback(name: str, args: dict):
        nonlocal confirmed
        confirmed = True
        return True, False

    mgr_ask = PermissionManager(
        mode=PermissionMode.ASK, confirmation_callback=mock_callback
    )
    allowed_ask, _ = await mgr_ask.evaluate_and_confirm(
        "run_command", {"command": "echo 1"}
    )
    assert confirmed is True
    assert allowed_ask is True
