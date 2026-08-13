"""Built-in agent tools (filesystem, terminal commands, web search)."""

import asyncio
import os
import subprocess
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


class ToolRegistry:
    def __init__(self):
        self.schemas: list[dict[str, Any]] = []
        self.handlers: dict[str, ToolHandler] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ToolHandler,
    ):
        self.schemas.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
        self.handlers[name] = handler

    async def execute(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.handlers:
            return f"Error: Unknown tool '{name}'"
        try:
            return await self.handlers[name](args)
        except Exception as e:
            return f"Error executing tool '{name}': {e!s}"


# Built-in Tool Implementations


async def view_file_handler(args: dict[str, Any]) -> str:
    path = args.get("path")
    if not path or not os.path.exists(path):
        return f"Error: File not found at path '{path}'"
    try:
        start_line = args.get("start_line", 1)
        end_line = args.get("end_line", 500)
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        selected_lines = lines[start_line - 1 : end_line]

        output = [
            f"--- File: {path} (Lines {start_line}-{min(end_line, total_lines)} of {total_lines}) ---"
        ]
        for idx, line in enumerate(selected_lines, start=start_line):
            output.append(f"{idx:4d} | {line.rstrip()}")
        return "\n".join(output)
    except Exception as e:
        return f"Error reading file: {e!s}"


async def edit_file_handler(args: dict[str, Any]) -> str:
    path = args.get("path")
    content = args.get("content", "")
    if not path:
        return "Error: Path is required"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully written to '{path}'"
    except Exception as e:
        return f"Error writing file: {e!s}"


from .patching import apply_patch_string, fuzzy_replace_string


async def replace_file_content_handler(args: dict[str, Any]) -> str:
    path = args.get("path")
    target = args.get("target_content")
    replacement = args.get("replacement_content")
    if not path or not os.path.exists(path):
        return f"Error: File not found at path '{path}'"
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()

        new_content, ok, mode = fuzzy_replace_string(content, target, replacement)
        if not ok:
            return f"Error: Target content not found in file '{path}' (fuzzy search failed)."

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully updated content in '{path}' ({mode})"
    except Exception as e:
        return f"Error modifying file: {e!s}"


async def apply_patch_handler(args: dict[str, Any]) -> str:
    patch = args.get("patch")
    if not patch:
        return "Error: 'patch' parameter is required."
    ok, msg = apply_patch_string(patch)
    if not ok:
        return f"❌ Patch failed: {msg}"
    return f"=== Unified Patch Applied ===\n{msg}"


async def list_dir_handler(args: dict[str, Any]) -> str:
    path = args.get("path", ".")
    if not os.path.exists(path):
        return f"Error: Directory not found at path '{path}'"
    try:
        entries = os.listdir(path)
        result = []
        for entry in sorted(entries):
            full_path = os.path.join(path, entry)
            is_dir = os.path.isdir(full_path)
            prefix = "[DIR] " if is_dir else "      "
            result.append(f"{prefix}{entry}")
        return "\n".join(result) if result else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {e!s}"


async def run_command_handler(args: dict[str, Any]) -> str:
    command = args.get("command")
    cwd = args.get("cwd", ".")
    if not command:
        return "Error: Command is required"
    try:
        process = await asyncio.create_subprocess_shell(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        out_str = stdout.decode("utf-8", errors="replace")
        err_str = stderr.decode("utf-8", errors="replace")

        res = f"Exit code: {process.returncode}\n"
        if out_str:
            res += f"--- STDOUT ---\n{out_str}\n"
        if err_str:
            res += f"--- STDERR ---\n{err_str}\n"
        return res
    except Exception as e:
        return f"Error running command: {e!s}"


async def read_url_handler(args: dict[str, Any]) -> str:
    url = args.get("url")
    if not url:
        return "Error: URL is required"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            resp = await client.get(url)
            return f"Status: {resp.status_code}\n\n{resp.text[:5000]}"
    except Exception as e:
        return f"Error fetching URL: {e!s}"


async def web_search_handler(args: dict[str, Any]) -> str:
    query = args.get("query")
    max_results = args.get("max_results", 5)
    if not query:
        return "Error: Search query is required"
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for query: '{query}'"

        output = [f"Search results for: '{query}'\n"]
        for idx, res in enumerate(results, 1):
            output.append(f"{idx}. {res.get('title', 'No Title')}")
            output.append(f"   URL: {res.get('href', '')}")
            output.append(f"   Snippet: {res.get('body', '')}\n")
        return "\n".join(output)
    except Exception as e:
        return f"Error executing web search: {e!s}"


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        "view_file",
        "View line-by-line contents of a file.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed)",
                },
                "end_line": {"type": "integer", "description": "Ending line number"},
            },
            "required": ["path"],
        },
        view_file_handler,
    )

    registry.register(
        "edit_file",
        "Write full content to a file (creates directories if missing).",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
        edit_file_handler,
    )

    registry.register(
        "replace_file_content",
        "Replace target text in a file with new content.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "target_content": {
                    "type": "string",
                    "description": "Exact text to find",
                },
                "replacement_content": {
                    "type": "string",
                    "description": "New text to insert",
                },
            },
            "required": ["path", "target_content", "replacement_content"],
        },
        replace_file_content_handler,
    )

    registry.register(
        "apply_patch",
        "Apply a unified diff patch string across single or multiple files in the workspace.",
        {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Unified diff patch text with ---/+++ headers and @@ hunks",
                },
            },
            "required": ["patch"],
        },
        apply_patch_handler,
    )

    registry.register(
        "list_dir",
        "List contents of a directory.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (defaults to current directory)",
                },
            },
            "required": [],
        },
        list_dir_handler,
    )

    registry.register(
        "run_command",
        "Run a shell command on the host system.",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command",
                },
            },
            "required": ["command"],
        },
        run_command_handler,
    )

    registry.register(
        "read_url",
        "Fetch raw content from a web URL.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL"},
            },
            "required": ["url"],
        },
        read_url_handler,
    )

    registry.register(
        "web_search",
        "Search the web using DuckDuckGo.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query keywords",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default 5)",
                },
            },
            "required": ["query"],
        },
        web_search_handler,
    )

    return registry


def register_skill_tools(registry: ToolRegistry, skill_manager: Any):
    async def load_skill_handler(args: dict[str, Any]) -> str:
        name = args.get("name")
        if not name or name not in skill_manager.skills:
            available = ", ".join(skill_manager.skills.keys()) or "None"
            return f"Error: Skill '{name}' not found. Available skills: {available}"

        skill = skill_manager.skills[name]
        out = [
            f"=== Loaded Skill: {skill.name} ===",
            f"Description: {skill.description}",
            f"File: {skill.location}\n",
            "--- Skill Instructions ---",
            skill.instructions,
        ]
        if skill.scripts:
            out.append("\n--- Helper Scripts ---")
            for s in skill.scripts:
                out.append(f"  • {s}")
        return "\n".join(out)

    registry.register(
        "load_skill",
        "Load detailed instructions and helper scripts for a specialized skill.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to load",
                },
            },
            "required": ["name"],
        },
        load_skill_handler,
    )

    return registry
