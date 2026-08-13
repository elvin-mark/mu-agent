import pytest

from mu_agent.tools import create_default_registry


@pytest.mark.asyncio
async def test_tool_registry_execution():
    registry = create_default_registry()

    # Test list_dir
    res = await registry.execute("list_dir", {"path": "."})
    assert "pyproject.toml" in res or "[DIR]" in res


@pytest.mark.asyncio
async def test_file_editing_and_viewing(tmp_path):
    registry = create_default_registry()
    test_file = str(tmp_path / "sample.txt")

    # Edit file
    res = await registry.execute(
        "edit_file", {"path": test_file, "content": "Hello World\nLine 2"}
    )
    assert "Successfully written" in res

    # View file
    res_view = await registry.execute(
        "view_file", {"path": test_file, "start_line": 1, "end_line": 2}
    )
    assert "Hello World" in res_view

    # Replace content
    res_rep = await registry.execute(
        "replace_file_content",
        {
            "path": test_file,
            "target_content": "Hello World",
            "replacement_content": "Hello Pi",
        },
    )
    assert "Successfully updated" in res_rep

    res_view2 = await registry.execute(
        "view_file", {"path": test_file, "start_line": 1, "end_line": 2}
    )
    assert "Hello Pi" in res_view2


@pytest.mark.asyncio
async def test_web_search_execution():
    registry = create_default_registry()
    res = await registry.execute(
        "web_search", {"query": "Python programming language", "max_results": 2}
    )
    assert "Search results for:" in res or "Python" in res
