"""Headless Codex runs with read-only shell access to the PR checkout."""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from .backend import LoginError, RunError, RunResult


class CodexError(RunError):
    provider = "Codex"


class CodexNotLoggedIn(CodexError, LoginError):
    pass


def codex_bin() -> str:
    return os.environ.get("PRELUDE_CODEX_BIN", "codex")


def _env() -> dict[str, str]:
    # Use saved CLI credentials, not an unrelated API key in PRelude's environment.
    stripped = {"GH_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"}
    return {k: v for k, v in os.environ.items() if k not in stripped and not k.startswith("ANTHROPIC_")}


def auth_status() -> dict:
    out = subprocess.run([codex_bin(), "login", "status"], capture_output=True, timeout=60, env=_env())
    return {"loggedIn": out.returncode == 0, "authMethod": "codex-cli"}


def ensure_logged_in() -> dict:
    status = auth_status()
    if not status["loggedIn"]:
        raise CodexNotLoggedIn("Codex is not logged in on this machine (run `codex login`).")
    return status


def _strict_schema(schema: dict) -> dict:
    """Structured Outputs requires optional object properties to be required and nullable."""
    schema = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            for value in node.values():
                visit(value)
            if "properties" in node:
                required = node.get("required", [])
                for key, value in node["properties"].items():
                    if key not in required:
                        node["properties"][key] = {"anyOf": [value, {"type": "null"}]}
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


def run_json(prompt: str, cwd: Path, *, schema: dict | None = None, **kwargs) -> RunResult:
    return _run(prompt, cwd, schema=schema, **kwargs)


def run_stream(prompt: str, cwd: Path, *, on_text: Callable[[str], None], **kwargs) -> RunResult:
    # The CLI emits completed assistant messages, rather than individual token deltas.
    return _run(prompt, cwd, on_text=on_text, **kwargs)


def _run(
    prompt: str,
    cwd: Path,
    *,
    schema: dict | None = None,
    model: str | None = None,
    effort: str | None = None,
    resume: str | None = None,
    fork: bool = False,
    timeout: int = 1500,
    on_text: Callable[[str], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
) -> RunResult:
    args = [codex_bin(), "exec", "--sandbox", "read-only", "--color", "never"]
    overrides = {
        "approval_policy": "never",
        "project_doc_max_bytes": 0,
        f"projects.{json.dumps(str(cwd.resolve()))}.trust_level": "untrusted",
        "features.apps": False,
        "features.plugins": False,
        "web_search": "disabled",
    }
    if effort:
        overrides["model_reasoning_effort"] = effort
    for key, value in overrides.items():
        args += ["-c", f"{key}={json.dumps(value)}"]
    if resume:
        args += ["fork" if fork else "resume", resume]
    args += ["--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--json"]
    if model:
        args += ["--model", model]

    with tempfile.TemporaryDirectory(prefix="prelude-run-") as folder:
        output = Path(folder) / "reply.txt"
        args += ["--output-last-message", str(output)]
        if schema is not None:
            schema_path = Path(folder) / "schema.json"
            schema_path.write_text(json.dumps(_strict_schema(schema)))
            args += ["--output-schema", str(schema_path)]
        args.append("-")
        # Files avoid blocking on a long prompt or a full stderr pipe while consuming stdout.
        with tempfile.TemporaryFile(mode="w+") as stdin, tempfile.TemporaryFile(mode="w+") as stderr:
            stdin.write(prompt)
            stdin.seek(0)
            proc = subprocess.Popen(
                args,
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=stderr,
                cwd=cwd,
                text=True,
                env=_env(),
                start_new_session=True,
            )
            expired = threading.Event()

            def kill():
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)

            def expire():
                expired.set()
                kill()

            timer = threading.Timer(timeout, expire)
            timer.start()
            session_id, complete, error, usage = None, False, None, None
            try:
                assert proc.stdout
                for line in proc.stdout:
                    event = json.loads(line)
                    kind = event.get("type")
                    if kind == "thread.started":
                        session_id = event["thread_id"]
                    elif kind == "turn.completed":
                        complete, usage = True, event.get("usage")
                    elif kind == "turn.failed":
                        error = (event.get("error") or {}).get("message", "turn failed")
                    elif kind == "error":
                        error = event.get("message", "unknown error")
                    elif kind in ("item.started", "item.completed"):
                        item = event.get("item") or {}
                        if kind == "item.completed" and item.get("type") == "agent_message" and on_text:
                            on_text("")
                            on_text(item.get("text", ""))
                        elif kind == "item.started" and on_activity:
                            on_activity(item.get("type", "tool").replace("_", " "))
                proc.wait()
            except json.JSONDecodeError as e:
                raise CodexError("Codex returned invalid JSON events") from e
            finally:
                timer.cancel()
                kill()
                proc.wait()
                if proc.stdout:
                    proc.stdout.close()
            if expired.is_set():
                raise CodexError(f"Codex run timed out after {timeout}s")
            if proc.returncode or not complete or not session_id:
                stderr.seek(0)
                detail = error or stderr.read()[-1500:] or "stream ended without a completed turn"
                raise CodexError(f"Codex run failed (exit {proc.returncode}): {detail}")
        if not output.exists():
            raise CodexError("Codex returned no final message")
        text = output.read_text().strip()
        try:
            structured = json.loads(text) if schema is not None else None
        except json.JSONDecodeError as e:
            raise CodexError("Codex returned no structured result") from e
        return RunResult(session_id, text, structured=structured, raw={"usage": usage})
