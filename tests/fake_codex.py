#!/usr/bin/env python3
"""Offline stand-in for Codex's JSONL protocol."""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

args = sys.argv[1:]
prompt = "" if args[:2] == ["login", "status"] else sys.stdin.read()
if log := os.environ.get("FAKE_CODEX_LOG"):
    schema = (
        json.loads(Path(args[args.index("--output-schema") + 1]).read_text()) if "--output-schema" in args else None
    )
    with open(log, "a") as f:
        f.write(json.dumps({"args": args, "env_keys": sorted(os.environ), "prompt": prompt, "schema": schema}) + "\n")
if args[:2] == ["login", "status"]:
    sys.exit(1 if os.environ.get("FAKE_CODEX_AUTH") == "none" else 0)


def emit(event):
    print(json.dumps(event), flush=True)


session = args[args.index("resume") + 1] if "resume" in args else str(uuid.uuid4())
emit({"type": "thread.started", "thread_id": session})
mode = os.environ.get("FAKE_CODEX_MODE")
if mode == "timeout":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(os.environ["FAKE_CODEX_CHILD"]).write_text(str(child.pid))
    time.sleep(60)
if mode == "malformed":
    print("not json", flush=True)
    sys.exit(0)
if mode == "failed":
    emit({"type": "turn.failed", "error": {"message": "usage limit reached"}})
    sys.exit(1)
if mode == "truncated":
    sys.exit(0)
reply = os.environ.get("FAKE_CODEX_REPLY", "Looks right.\nVERDICT: confirmed")
if "--output-schema" in args:
    reply = os.environ.get("FAKE_CODEX_STRUCTURED", '{"summary":"ok","findings":[]}')
emit({"type": "item.started", "item": {"type": "command_execution", "id": "1"}})
emit({"type": "item.completed", "item": {"type": "agent_message", "id": "2", "text": reply}})
Path(args[args.index("--output-last-message") + 1]).write_text(reply)
emit({"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 20}})
