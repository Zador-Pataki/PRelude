import json
import time
from pathlib import Path

import pytest

from prelude import backend, chat, cli, codex, discuss, memory_agent, review
from prelude.config import load_config
from prelude.github import PrRef
from prelude.memory import Memory
from prelude.prompts import LEARN_SCHEMA, REVIEW_SCHEMA
from prelude.workspace import Checkout

from .conftest import DIFF, FINDING
from .test_review import FakeReader


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_config_and_auth(cfg, monkeypatch, capsys):
    path = cfg.data_dir / "config.yaml"
    path.write_text("backend: codex\ncodex: {model: review-model, effort: high, reviewers: 3}\n")
    cfg = load_config(cfg.data_dir)
    assert cfg.agent.model == "review-model" and cfg.agent.reviewers == 3 and cfg.agent.effort == "high"
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    assert cli.main(["auth"]) == 0
    assert "codex" in capsys.readouterr().out
    monkeypatch.setenv("FAKE_CODEX_AUTH", "none")
    with pytest.raises(backend.LoginError, match="codex login"):
        backend.ensure_logged_in(cfg)
    path.write_text("backend: unknown\n")
    with pytest.raises(ValueError, match="backend"):
        load_config(cfg.data_dir)


def test_structured_output_and_restrictions(fake_codex, monkeypatch, tmp_path):
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY"):
        monkeypatch.setenv(key, "unrelated-secret")
    result = codex.run_json("review", tmp_path, schema=REVIEW_SCHEMA, model="review-model", effort="high")
    assert result.structured == {"summary": "ok", "findings": []} and result.cost_usd is None
    call = calls(fake_codex)[0]
    args = call["args"]
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert {"--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"} <= set(args)
    assert 'approval_policy="never"' in args and 'model_reasoning_effort="high"' in args
    assert "features.apps=false" in args and "features.plugins=false" in args
    assert not {"GH_TOKEN", "GITHUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY"} & set(
        call["env_keys"]
    )
    item = call["schema"]["properties"]["findings"]["items"]
    assert "start_line" in item["required"] and "anyOf" in item["properties"]["start_line"]
    assert "start_line" not in REVIEW_SCHEMA["properties"]["findings"]["items"]["required"]


@pytest.mark.parametrize(
    "mode,message", [("failed", "usage limit"), ("truncated", "completed turn"), ("malformed", "JSON")]
)
def test_failures_are_not_success(mode, message, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)
    with pytest.raises(codex.CodexError, match=message):
        codex.run_json("review", tmp_path)


def test_timeout_kills_child_processes(monkeypatch, tmp_path):
    pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_CODEX_MODE", "timeout")
    monkeypatch.setenv("FAKE_CODEX_CHILD", str(pid_file))
    with pytest.raises(codex.CodexError, match="timed out"):
        codex.run_json("review", tmp_path, timeout=1)
    state = Path(f"/proc/{pid_file.read_text()}/stat")
    for _ in range(50):
        if not state.exists() or state.read_text().split()[2] == "Z":
            return
        time.sleep(0.02)
    pytest.fail("runner left its child alive")


def test_panel_discussions_and_backend_switch(cfg, store, fake_codex, fake_claude, monkeypatch, tmp_path):
    cfg.backend = "codex"
    cfg.codex.model = "review-model"
    monkeypatch.setattr(review, "prepare_checkout", lambda cfg, pr: Checkout(tmp_path, "c" * 40, DIFF, ["src/auth.ts"]))
    monkeypatch.setenv("FAKE_CODEX_STRUCTURED", json.dumps({"summary": "ok", "findings": [dict(FINDING, found_by=2)]}))
    rid = review.start_review(store, cfg, PrRef("B1ueber2y", "PRelude", 1), reviewers=2)
    cfg.backend = "claude"  # Queued work and replies keep their original backend.
    review.run_review(store, cfg, rid, github=FakeReader())
    row = store.get_review(rid)
    assert row["status"] == "done" and row["backend"] == "codex" and row["cost_usd"] is None
    assert len(json.loads(row["reports"])) == 2
    assert len([c for c in calls(fake_codex) if c["args"][0] == "exec"]) == 3
    finding = store.findings_for(rid)[0]
    broker = discuss.Broker()
    for _ in range(2):
        t = discuss.post_message(store, cfg, broker, finding["id"], "Why?", ping=False)
        t.join(5)
        assert not t.is_alive()
    runs = [c for c in calls(fake_codex) if c["args"][0] == "exec"]
    assert "fork" in runs[-2]["args"] and "resume" in runs[-1]["args"]
    assert len([m for m in store.messages_for(finding["id"]) if m["role"] == "assistant"]) == 2
    cid = store.create_chat(row)
    t = chat.post_message(store, cfg, broker, cid, "Explain the PR")
    t.join(5)
    assert store.get_chat(cid)["session_id"] and store.get_chat(cid)["backend"] == "codex"
    assert not fake_claude.exists()


def test_existing_claude_sessions_stay_claude(cfg, store, done_review, fake_codex, fake_claude):
    cfg.backend = "codex"
    cid = store.create_chat(store.get_review(done_review))
    t = chat.post_message(store, cfg, discuss.Broker(), cid, "Why?")
    t.join(5)
    assert store.get_chat(cid)["session_id"] and not fake_codex.exists()
    assert "--fork-session" in calls(fake_claude)[-1]["args"]


def test_memory_uses_selected_backend(cfg, monkeypatch, fake_codex, fake_claude):
    cfg.backend = "codex"
    monkeypatch.setenv("FAKE_CODEX_STRUCTURED", '{"lesson":null,"scope":"global"}')
    assert memory_agent._run(cfg, Memory(cfg), "Remember?", LEARN_SCHEMA)["lesson"] is None
    assert calls(fake_codex) and not fake_claude.exists()
