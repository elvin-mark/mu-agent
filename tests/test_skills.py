from mu_agent.skills import SkillManager, parse_skill_md


def test_parse_skill_md(tmp_path):
    skill_dir = tmp_path / "pdf-parser"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: pdf-parser
description: Parse PDF forms and extract tables
---

# PDF Parser Skill Instructions
1. Use pdfplumber to extract text.
2. Format output as JSON.
""",
        encoding="utf-8",
    )

    skill = parse_skill_md(str(skill_file))
    assert skill is not None
    assert skill.name == "pdf-parser"
    assert skill.description == "Parse PDF forms and extract tables"
    assert "pdfplumber" in skill.instructions


def test_skill_manager_discovery(tmp_path):
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "python-refactor"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: python-refactor
description: Automated Python refactoring workflow
---
Refactor instructions here.
""",
        encoding="utf-8",
    )

    manager = SkillManager(search_paths=[str(skills_root)])
    assert "python-refactor" in manager.skills
    prompt = manager.get_skill_summary_prompt()
    assert "python-refactor" in prompt
