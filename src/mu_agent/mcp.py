"""MCP (Model Context Protocol) Client Integration (.mu/mcp.json)."""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.abspath(".mu/mcp.json")
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.load_config()

    def load_config(self):
        """Load MCP server definitions from .mu/mcp.json."""
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.servers = data.get("mcpServers", {})
        except Exception:
            pass

    async def connect_and_register_tools(self, tool_registry):
        """Connect to configured MCP servers via STDIO and register tools."""
        if not self.servers:
            return

        for name, config in self.servers.items():
            command = config.get("command")
            args = config.get("args", [])
            env = config.get("env", None)
            if not command:
                continue

            try:
                server_params = StdioServerParameters(
                    command=command, args=args, env=env
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()

                        for tool in tools_result.tools:
                            mcp_tool_name = f"mcp_{name}_{tool.name}"

                            def create_handler(srv_name, t_name, cmd, a, e):
                                async def handler(tool_args: Dict[str, Any]) -> str:
                                    params = StdioServerParameters(
                                        command=cmd, args=a, env=e
                                    )
                                    async with stdio_client(params) as (r, w):
                                        async with ClientSession(r, w) as sess:
                                            await sess.initialize()
                                            res = await sess.call_tool(
                                                t_name, tool_args
                                            )
                                            out_lines = []
                                            for content in res.content:
                                                if hasattr(content, "text"):
                                                    out_lines.append(content.text)
                                            return (
                                                "\n".join(out_lines)
                                                if out_lines
                                                else "Tool executed with no text output."
                                            )

                                return handler

                            handler_func = create_handler(
                                name, tool.name, command, args, env
                            )
                            schema_params = (
                                tool.inputSchema
                                if hasattr(tool, "inputSchema")
                                else {"type": "object", "properties": {}}
                            )

                            tool_registry.register(
                                name=mcp_tool_name,
                                description=f"[MCP {name}] {tool.description or ''}",
                                parameters=schema_params,
                                handler=handler_func,
                            )
            except Exception:
                pass
