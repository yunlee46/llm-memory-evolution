"""The coding agent: a multi-turn function-calling loop over a sandboxed workspace.

Fairness note, and it is the load-bearing detail of the whole experiment:
both groups receive the *same* BASE_SYSTEM describing the harness mechanics
(tools available, what to produce, how to stop). The memory file is appended
for group A only. Harness mechanics are not "memory", so putting them in the
shared base is what makes the memory file the single difference between the
groups — and therefore what makes `lift_g` attributable to it (AGENTS.md §1).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from .config import ExperimentConfig
from .evaluator import Evaluator
from .llm import GeminiClient, TokenUsage
from .sandbox import SandboxError, Workspace
from .tasks import Task

logger = logging.getLogger(__name__)

BASE_SYSTEM = """\
You are a software engineer working in a sandboxed directory.

Produce a working solution to the task you are given by calling the tools
provided. Rules of the environment:

- Your working directory starts EMPTY. Do not list or read files before your
  first write — there is nothing there yet.
- Your first action should be `write_file` with the complete solution. Do not
  write a skeleton and fill it in later; you have very few turns.
- You can only read and write files inside your working directory.
- The test suite is NOT in your working directory and you cannot read or edit
  it. You may run it with `run_tests` and see the results.
- Your turn budget is small. Spend it on: write the full solution, run the
  tests, fix what failed. Calling `run_tests` before you have written anything
  wastes a turn.
- When you believe the task is complete, call `finish` with a short summary.
- If you run out of turns without calling `finish`, whatever you have written
  is still scored as it stands.
"""

MEMORY_PREAMBLE = """\

---

The following notes were accumulated by previous agents working on this task.
Use them as you see fit.

"""


@dataclass
class AgentResult:
    group: str
    finished: bool
    stop_reason: str
    turns: int
    usage: TokenUsage
    wall_clock_s: float
    summary: str = ""
    transcript: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "finished": self.finished,
            "stop_reason": self.stop_reason,
            "turns": self.turns,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "summary": self.summary,
            "violations": self.violations,
            **self.usage.as_dict(),
        }


def build_tool_declarations(include_run_tests: bool) -> types.Tool:
    declarations = [
        types.FunctionDeclaration(
            name="list_files",
            description="List all files in your working directory.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="read_file",
            description="Read a file from your working directory.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "path": types.Schema(
                        type="STRING", description="Path relative to the working directory."
                    )
                },
                required=["path"],
            ),
        ),
        types.FunctionDeclaration(
            name="write_file",
            description="Write a file to your working directory, overwriting it if present.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "path": types.Schema(
                        type="STRING", description="Path relative to the working directory."
                    ),
                    "content": types.Schema(
                        type="STRING", description="Full contents of the file."
                    ),
                },
                required=["path", "content"],
            ),
        ),
        types.FunctionDeclaration(
            name="finish",
            description="Declare the task complete. Call this when you are done.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "summary": types.Schema(
                        type="STRING", description="One or two sentences on what you built."
                    )
                },
                required=["summary"],
            ),
        ),
    ]
    if include_run_tests:
        declarations.insert(
            3,
            types.FunctionDeclaration(
                name="run_tests",
                description=(
                    "Run the task's test suite against your current files and "
                    "return the results."
                ),
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
        )
    return types.Tool(function_declarations=declarations)


class CodingAgent:
    """One agent attempt. Owns no state between runs — the workspace is passed in."""

    def __init__(
        self,
        client: GeminiClient,
        config: ExperimentConfig,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.evaluator = evaluator

    async def run(
        self,
        task: Task,
        workspace: Workspace,
        *,
        memory: str | None = None,
        group: str = "A",
        seed: int | None = None,
    ) -> AgentResult:
        sandbox_cfg = self.config.sandbox
        system_instruction = BASE_SYSTEM
        if memory:
            system_instruction += MEMORY_PREAMBLE + memory

        tools = [build_tool_declarations(include_run_tests=self.evaluator is not None)]
        history: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=task.prompt())])
        ]

        transcript: list[dict[str, Any]] = []
        violations: list[str] = []
        usage = TokenUsage()
        started = time.monotonic()
        summary = ""
        stop_reason = "max_turns"
        finished = False
        turn = 0

        while turn < sandbox_cfg.max_turns:
            elapsed = time.monotonic() - started
            if elapsed > sandbox_cfg.wall_clock_s:
                stop_reason = "wall_clock"
                break
            if usage.total > sandbox_cfg.max_tokens_per_run:
                stop_reason = "token_cap"
                break

            turn += 1
            response = await self.client.generate(
                model=self.config.models.worker,
                contents=history,
                role="worker",
                system_instruction=system_instruction,
                temperature=self.config.models.worker_temperature,
                seed=seed,
                thinking_budget=self.config.models.worker_thinking_budget,
                tools=tools,
            )
            usage = usage + response.usage

            model_content = self._model_content(response)
            if model_content is not None:
                # Appended verbatim rather than reconstructed: Gemini 3 models
                # carry thought signatures on their parts, and dropping them
                # breaks multi-turn function calling.
                history.append(model_content)

            if response.text:
                transcript.append({"turn": turn, "type": "model_text", "text": response.text})

            calls = response.function_calls
            if not calls:
                stop_reason = "no_tool_call"
                summary = summary or (response.text or "").strip()
                break

            response_parts: list[types.Part] = []
            for call in calls:
                args = dict(call.args or {})
                transcript.append(
                    {"turn": turn, "type": "tool_call", "name": call.name, "args": _redact(args)}
                )

                if call.name == "finish":
                    summary = str(args.get("summary", ""))
                    finished = True
                    stop_reason = "finished"
                    break

                result = self._dispatch(call.name, args, workspace, task, violations)
                transcript.append(
                    {
                        "turn": turn,
                        "type": "tool_result",
                        "name": call.name,
                        "result": _truncate(result),
                    }
                )
                response_parts.append(
                    types.Part.from_function_response(name=call.name, response={"result": result})
                )

            if finished:
                break
            if response_parts:
                history.append(types.Content(role="user", parts=response_parts))

        return AgentResult(
            group=group,
            finished=finished,
            stop_reason=stop_reason,
            turns=turn,
            usage=usage,
            wall_clock_s=time.monotonic() - started,
            summary=summary,
            transcript=transcript,
            violations=violations,
        )

    # ------------------------------------------------------------- dispatch

    def _dispatch(
        self,
        name: str,
        args: dict[str, Any],
        workspace: Workspace,
        task: Task,
        violations: list[str],
    ) -> Any:
        """Execute one tool call. Never raises: the model sees errors as results."""
        try:
            if name == "list_files":
                return workspace.list_files()

            if name == "read_file":
                return workspace.read_file(str(args.get("path", "")))

            if name == "write_file":
                path = str(args.get("path", ""))
                if self._targets_tests(path, task):
                    violations.append(f"write_file into test path: {path}")
                workspace.write_file(path, str(args.get("content", "")))
                return f"wrote {path}"

            if name == "run_tests":
                return self._run_tests(workspace, task)

            violations.append(f"unknown tool: {name}")
            return f"error: unknown tool {name!r}"

        except SandboxError as exc:
            violations.append(f"{name}: {exc}")
            return f"error: {exc}"
        except FileNotFoundError:
            return f"error: no such file: {args.get('path')!r}"
        except OSError as exc:
            return f"error: {exc}"

    @staticmethod
    def _targets_tests(path: str, task: Task) -> bool:
        parts = path.replace("\\", "/").split("/")
        return task.tests_dir.name in parts or task.held_out_dir.name in parts

    def _run_tests(self, workspace: Workspace, task: Task) -> dict[str, Any]:
        """Run the visible suite only. Held-out results are never shown."""
        assert self.evaluator is not None
        result = self.evaluator.evaluate(
            workspace, task, include_held_out=False, network=self.config.sandbox.network
        )
        return {
            "passed": result.visible.passed,
            "total": result.visible.expected,
            "output": _truncate(result.visible.stdout or result.visible.stderr, 4000),
        }

    @staticmethod
    def _model_content(response: Any) -> types.Content | None:
        candidates = getattr(response.raw, "candidates", None) or []
        return candidates[0].content if candidates else None


def _truncate(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"\n[...truncated, {len(value) - limit} chars omitted]"
    return value


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Keep transcripts readable — file bodies are recoverable from the workspace."""
    return {
        key: (_truncate(value, 500) if key == "content" else value)
        for key, value in args.items()
    }


def transcript_to_json(result: AgentResult) -> str:
    return json.dumps(
        {"meta": result.as_dict(), "transcript": result.transcript}, indent=2
    )
