"""Shared runner interface for reviews and discussions."""

from dataclasses import dataclass, field
from typing import Any

from .config import Config


class RunError(RuntimeError):
    provider = "Reviewer"

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


class LoginError(RunError):
    pass


@dataclass
class RunResult:
    session_id: str
    text: str
    structured: Any = None
    cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _runner(cfg: Config):
    if cfg.backend == "claude":
        from . import claude

        return claude, {"auth": claude.auth_for(cfg)}
    if cfg.backend == "codex":
        from . import codex

        return codex, {}
    raise ValueError(f"Unknown review backend: {cfg.backend!r}")


def auth_status(cfg: Config) -> dict:
    runner, kwargs = _runner(cfg)
    return runner.auth_status(**kwargs)


def ensure_logged_in(cfg: Config) -> dict:
    runner, kwargs = _runner(cfg)
    return runner.ensure_logged_in(**kwargs)


def run_json(cfg: Config, *args, **kwargs) -> RunResult:
    runner, auth = _runner(cfg)
    return runner.run_json(*args, **kwargs, **auth)


def run_stream(cfg: Config, *args, **kwargs) -> RunResult:
    runner, auth = _runner(cfg)
    return runner.run_stream(*args, **kwargs, **auth)
