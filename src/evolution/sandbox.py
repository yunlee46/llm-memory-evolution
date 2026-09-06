"""Isolated per-run workspaces and command execution.

Two guarantees matter here (AGENTS.md §4):

1. Every run starts from a clean workspace. A group-B agent that inherits state
   from a previous run silently corrupts the experiment's only baseline.
2. The agent can never touch the tests. This is enforced structurally: the
   workspace does not contain them. The evaluator assembles a *separate*
   scoring directory from workspace + tests at scoring time, so there is no
   path an agent could write to even if it tried.

Backends:
  docker - real isolation, enforces `--network none`.
  local  - subprocess in a temp dir. Cannot enforce network denial; warns.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import SandboxConfig

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_IMAGE = "llm-memory-evolution-runner:latest"


class SandboxError(RuntimeError):
    pass


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Workspace:
    """A clean directory the agent may write to. Contains no tests, ever."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, scaffold: Path | None = None, prefix: str = "lme-ws-") -> Workspace:
        root = Path(tempfile.mkdtemp(prefix=prefix))
        if scaffold is not None:
            if not scaffold.exists():
                raise SandboxError(f"scaffold directory not found: {scaffold}")
            shutil.copytree(scaffold, root, dirs_exist_ok=True)
        return cls(root)

    def write_file(self, relative_path: str, content: str) -> Path:
        target = self._resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def read_file(self, relative_path: str) -> str:
        return self._resolve(relative_path).read_text(encoding="utf-8")

    def list_files(self) -> list[str]:
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )

    def _resolve(self, relative_path: str) -> Path:
        """Resolve a path, refusing anything that escapes the workspace."""
        candidate = (self.root / relative_path).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise SandboxError(
                f"path escapes workspace: {relative_path!r}"
            )
        return candidate

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()


class SandboxBackend(Protocol):
    name: str

    def run(
        self, command: list[str], cwd: Path, timeout_s: int, network: str
    ) -> ExecResult: ...


class LocalBackend:
    """Subprocess execution. No network isolation — development use only."""

    name = "local"

    def run(
        self, command: list[str], cwd: Path, timeout_s: int, network: str
    ) -> ExecResult:
        if network == "deny":
            logger.warning(
                "sandbox.backend=local cannot enforce network=deny; "
                "agent-generated code has network access. Use the docker "
                "backend for real runs."
            )
        # "python" means the container's interpreter under docker; locally it
        # must mean *this* interpreter, or we'd run against whatever is on PATH.
        if command and command[0] == "python":
            command = [sys.executable, *command[1:]]
        try:
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\n[timeout after {timeout_s}s]",
                timed_out=True,
            )
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)


class DockerBackend:
    """Container execution with network denial and a non-root user."""

    name = "docker"

    def __init__(self, image: str) -> None:
        self.image = image

    def run(
        self, command: list[str], cwd: Path, timeout_s: int, network: str
    ) -> ExecResult:
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none" if network == "deny" else "bridge",
            "--memory",
            "1g",
            "--cpus",
            "2",
            "-v",
            f"{cwd.resolve()}:/work",
            "-w",
            "/work",
            self.image,
            *command,
        ]
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s + 30,  # container startup overhead
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\n[timeout after {timeout_s}s]",
                timed_out=True,
            )
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)


def docker_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def select_backend(config: SandboxConfig) -> SandboxBackend:
    """Resolve the configured backend, falling back loudly if docker is down."""
    if config.backend == "local":
        return LocalBackend()
    if config.backend == "docker":
        if not docker_available():
            raise SandboxError(
                "sandbox.backend=docker but the Docker daemon is not reachable. "
                "Start Docker Desktop, or set sandbox.backend=local (no network "
                "isolation)."
            )
        return DockerBackend(config.image)

    if docker_available():
        return DockerBackend(config.image)
    logger.warning(
        "Docker daemon unreachable; falling back to the local backend. "
        "Network denial is NOT enforced. Fine for development, not for a "
        "scored run."
    )
    return LocalBackend()
