from mu_agent.session import SessionManager, load_project_instructions
from mu_agent.types import Message, Role


def test_session_manager(tmp_path):
    sessions_dir = str(tmp_path / "sessions")
    mgr = SessionManager(session_id="test1234", sessions_dir=sessions_dir)
    msg1 = Message(role=Role.USER, content="Hello Pi")
    mgr.save_message(msg1)

    loaded = mgr.load_session()
    assert len(loaded) == 1
    assert loaded[0].content == "Hello Pi"


def test_project_instructions(tmp_path):
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("Do not use external APIs.", encoding="utf-8")

    instructions = load_project_instructions(root_dir=str(tmp_path))
    assert "Do not use external APIs." in instructions
