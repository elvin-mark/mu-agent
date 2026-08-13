"""Skill Discovery & Management System (.mu/skills/ and ~/.mu/skills/)."""

import glob
import os
import re

from pydantic import BaseModel


class Skill(BaseModel):
    name: str
    description: str
    location: str  # Path to SKILL.md
    instructions: str
    scripts: list[str] = []


def parse_skill_md(file_path: str) -> Skill | None:
    """Parse SKILL.md file with YAML-like frontmatter metadata."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse YAML frontmatter between --- and ---
        name = os.path.basename(os.path.dirname(file_path)) or "unknown"
        description = "No description provided."
        instructions = content

        frontmatter_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL
        )
        if frontmatter_match:
            yaml_block = frontmatter_match.group(1)
            instructions = frontmatter_match.group(2).strip()

            for line in yaml_block.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip().lower()
                    val = val.strip().strip("'\"")
                    if key == "name":
                        name = val
                    elif key == "description":
                        description = val

        # Discover helper scripts if any
        skill_dir = os.path.dirname(file_path)
        scripts_dir = os.path.join(skill_dir, "scripts")
        scripts = []
        if os.path.exists(scripts_dir):
            scripts = [os.path.join(scripts_dir, s) for s in os.listdir(scripts_dir)]

        return Skill(
            name=name,
            description=description,
            location=file_path,
            instructions=instructions,
            scripts=scripts,
        )
    except Exception:
        return None


class SkillManager:
    def __init__(self, search_paths: list[str] | None = None):
        if search_paths is None:
            search_paths = [
                os.path.abspath(".mu/skills"),
                os.path.abspath(".pi/skills"),
                os.path.expanduser("~/.mu/skills"),
                os.path.expanduser("~/.pi/skills"),
            ]
        self.search_paths = search_paths
        self.skills: dict[str, Skill] = {}
        self.scan_skills()

    def scan_skills(self):
        """Discover skills in search paths."""
        self.skills.clear()
        for base_path in self.search_paths:
            if not os.path.exists(base_path):
                continue
            pattern = os.path.join(base_path, "*", "SKILL.md")
            for skill_file in glob.glob(pattern):
                skill = parse_skill_md(skill_file)
                if skill and skill.name not in self.skills:
                    self.skills[skill.name] = skill

    def get_skill_summary_prompt(self) -> str:
        """Generate prompt metadata listing available skills."""
        if not self.skills:
            return ""
        lines = ["Available Specialized Skills:"]
        for skill in self.skills.values():
            lines.append(
                f"- **{skill.name}**: {skill.description} (Location: {skill.location})"
            )
        lines.append(
            "\nUse the `load_skill` tool or `/skill <name>` to load instructions for a skill when relevant."
        )
        return "\n".join(lines)
