<p align="center">
  <h1 align="center">⚡ Mu Agent</h1>
  <p align="center">
    <strong>An extensible, multi-provider AI coding agent built in Python and managed with <code>uv</code>.</strong>
  </p>
</p>

---

## 🌟 Overview

**Mu Agent** (`mu-agent`) is a high-performance Python port and evolution of the Pi agent harness. It features an interactive terminal UI built with **Textual** & **Rich**, native support for cloud (OpenAI, Anthropic, Gemini) and local models (Ollama, LM Studio, vLLM), session persistence, dynamic tool execution, skill discovery, MCP (Model Context Protocol) integration, and subagent delegation.

---

## ✨ Features

- **💻 Rich Interactive TUI**: Terminal Interface built with `textual` & `rich` featuring live real-time token telemetry, speed (tok/s), latency metrics, and wrap-formatted chat logs.
- **⚡ Unified Multi-Provider LLM Engine**:
  - Cloud Providers: OpenAI, Anthropic, Google Gemini.
  - Local LLMs: Ollama (`--provider ollama`), LM Studio, vLLM, and any OpenAI-compatible base URL (`--base-url`).
- **🛠️ Built-in Tool Suite**:
  - `view_file` & `edit_file`: Line-by-line inspection and writing.
  - `replace_file_content`: Precise string replacements.
  - `list_dir`: Directory inspection.
  - `run_command`: Async terminal process execution.
  - `web_search`: Live search via DuckDuckGo (`ddgs`).
  - `read_url`: Fetch web page contents.
- **🧰 Skill System (`.mu/skills/`)**:
  - Automatically discovers skill packages (`SKILL.md`) in `.mu/skills/` or `~/.mu/skills/`.
  - Registered in system prompts and invokable via LLM function calling (`load_skill`) or slash commands (`/skills`, `/skill <name>`).
- **🔌 Model Context Protocol (MCP)**:
  - Connects to external MCP servers (STDIO) configured in `.mu/mcp.json`.
  - Dynamically registers external tools (e.g. SQLite, GitHub, Brave Search) into Mu's tool engine.
- **🤖 Subagent Swarm & Messaging (`spawn_subagent`, `send_subagent_message`, `get_subagent_status`)**:
  - Allows Mu to spawn subagents and pass messages asynchronously between parent and peer subagents.
- **🪝 Lifecycle Hook & Plugin System (`.mu/plugins/`)**:
  - Auto-discovers Python scripts in `.mu/plugins/` with `@hook` decorators for `pre_tool_call`, `post_tool_call`, `on_turn_start`, `on_turn_end`, and `on_context_compact`.
- **🧠 Multi-Tier Context Compaction**:
  - Automated 2-tier context budget management: output truncation at 70% budget, followed by LLM-driven semantic summarization pass at 90% capacity.
- **📑 Session Persistence & Resuming (`.mu/sessions/`)**:
  - All turns and tool outputs are saved as JSONL records in `.mu/sessions/`.
  - Resume any past session via `--session <session_id>`.
- **💡 Rich Slash Commands**:
  - `/stats`, `/clear`, `/session`, `/tools`, `/skills`, `/skill <name>`, `/model [name]`, `/compact`, `/system`, `/export`, `/help`.
- **📦 Zero-Dependency Single Binary**:
  - Includes a PyInstaller script to compile `mu-agent` into a standalone, portable binary executable (`dist/mu`).

---

## 🚀 Quick Start

### Prerequisites
Ensure you have [uv](https://github.com/astral-sh/uv) installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation & Execution

```bash
# Clone the repository
git clone https://github.com/elvin-mark/mu-agent.git
cd mu-agent/pi-agent-py

# Run with OpenAI (default)
OPENAI_API_KEY="your-api-key" uv run mu --provider openai --model gpt-4o

# Run with Anthropic
ANTHROPIC_API_KEY="your-api-key" uv run mu --provider anthropic --model claude-3-5-sonnet-20241022

# Run with local Ollama
uv run mu --provider ollama --model qwen2.5-coder

# Run with custom local server (LM Studio, vLLM, etc.)
uv run mu --provider openai --base-url http://localhost:1234/v1 --model local-model
```

---

## 💡 Slash Commands

Inside the interactive terminal interface, start your input with `/` to run built-in commands:

| Command | Description |
|---|---|
| `/stats` | View token usage (prompt, completion, total), speed (tok/s), and latency. |
| `/model [name]` | View active LLM provider/model or switch models on the fly (e.g. `/model qwen2.5-coder`). |
| `/skills` | List all discovered skills in `.mu/skills/` or `~/.mu/skills/`. |
| `/skill <name>` | Inspect detailed instructions and helper scripts for a specific skill. |
| `/tools` | List all registered built-in, skill, and MCP tools. |
| `/session` | View current session ID and JSONL session log file path. |
| `/compact` | Manually trigger context compaction and truncate old tool logs. |
| `/system` | View active system prompt and loaded project instructions (`AGENTS.md`). |
| `/export` | Export the session transcript to a clean Markdown document. |
| `/clear` | Clear the terminal chat log window. |
| `/help` | List all available slash commands. |

---

## 🧰 Skills Layout (`.mu/skills/`)

Define custom domain-specific skills by placing a folder with a `SKILL.md` file in `.mu/skills/`:

```
.mu/skills/
└── pdf-parser/
    ├── SKILL.md
    └── scripts/
        └── extract.py
```

*Example `SKILL.md`*:
```markdown
---
name: pdf-parser
description: Parse PDF forms and extract structured JSON tables
---

# PDF Parser Instructions
1. Use pdfplumber or pypdf to extract form text.
2. Format extracted tabular data into JSON format.
```

---

## 🔌 Model Context Protocol (MCP) Configuration (`.mu/mcp.json`)

Configure external MCP servers in `.mu/mcp.json`:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "app.db"]
    }
  }
}
```

Mu Agent will automatically discover and register tools exposed by the MCP server on launch!

---

## 📦 Building Standalone Executable Binary

To package `mu-agent` into a standalone, single-file binary executable with zero external dependencies:

```bash
uv run python scripts/build_binary.py
```

The compiled binary will be placed at **`./dist/mu`**. You can move `./dist/mu` to `/usr/local/bin` and run it on any machine without Python or `uv`!

---

## 🧪 Running Tests

```bash
# Run unit test suite
uv run pytest

# Check formatting and linting
uv run ruff format --check .
uv run ruff check .
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
