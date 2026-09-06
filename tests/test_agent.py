"""Agent loop behaviour, with a fake client — no network calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from evolution.agent import (
    BASE_SYSTEM,
    CodingAgent,
    build_tool_declarations,
)
from evolution.config import load_config
from evolution.llm import LLMResponse, TokenUsage
from evolution.sandbox import Workspace
from evolution.tasks import TaskRegistry
from google.genai import types


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    cfg = load_config()
    cfg.sandbox.max_turns = 4
    return cfg


@pytest.fixture(scope="module")
def task():
    return TaskRegistry.load().get("word_freq")


def _response(text: str = "", calls: list | None = None) -> LLMResponse:
    content = types.Content(role="model", parts=[types.Part(text=text or "...")])
    return LLMResponse(
        text=text,
        usage=TokenUsage(prompt=10, output=5),
        model_id="fake",
        function_calls=calls or [],
        raw=SimpleNamespace(candidates=[SimpleNamespace(content=content)]),
    )


def _call(name: str, **args):
    return SimpleNamespace(name=name, args=args)


class FakeClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.kwargs: list[dict] = []

    async def generate(self, **kwargs):
        self.kwargs.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return _response(text="out of scripted responses")


class TestGroupFairness:
    """Memory must be the ONLY difference between the groups (AGENTS.md §1)."""

    @pytest.mark.asyncio
    async def test_group_b_gets_base_system_only(self, config, task):
        client = FakeClient([_response(calls=[_call("finish", summary="done")])])
        agent = CodingAgent(client, config)

        with Workspace.create() as workspace:
            await agent.run(task, workspace, memory=None, group="B")

        assert client.kwargs[0]["system_instruction"] == BASE_SYSTEM

    @pytest.mark.asyncio
    async def test_group_a_gets_base_system_plus_memory(self, config, task):
        client = FakeClient([_response(calls=[_call("finish", summary="done")])])
        agent = CodingAgent(client, config)

        with Workspace.create() as workspace:
            await agent.run(task, workspace, memory="P-001 Write tests first.", group="A")

        system = client.kwargs[0]["system_instruction"]
        assert system.startswith(BASE_SYSTEM)
        assert "P-001 Write tests first." in system

    @pytest.mark.asyncio
    async def test_both_groups_receive_the_identical_task_prompt(self, config, task):
        prompts = []
        for memory in (None, "some memory"):
            client = FakeClient([_response(calls=[_call("finish", summary="d")])])
            with Workspace.create() as workspace:
                await CodingAgent(client, config).run(task, workspace, memory=memory)
            prompts.append(client.kwargs[0]["contents"][0].parts[0].text)

        assert prompts[0] == prompts[1] == task.prompt()


class TestToolDeclarations:
    def test_run_tests_offered_when_evaluator_present(self):
        names = {d.name for d in build_tool_declarations(True).function_declarations}
        assert names == {"list_files", "read_file", "write_file", "run_tests", "finish"}

    def test_run_tests_withheld_when_no_evaluator(self):
        names = {d.name for d in build_tool_declarations(False).function_declarations}
        assert "run_tests" not in names


class TestLoopTermination:
    @pytest.mark.asyncio
    async def test_finish_ends_the_run(self, config, task):
        client = FakeClient(
            [
                _response(calls=[_call("write_file", path="word_freq.py", content="x=1")]),
                _response(calls=[_call("finish", summary="built it")]),
            ]
        )
        with Workspace.create() as workspace:
            result = await CodingAgent(client, config).run(task, workspace)

        assert result.finished
        assert result.stop_reason == "finished"
        assert result.summary == "built it"
        assert result.turns == 2

    @pytest.mark.asyncio
    async def test_max_turns_caps_a_looping_agent(self, config, task):
        client = FakeClient([_response(calls=[_call("list_files")]) for _ in range(20)])
        with Workspace.create() as workspace:
            result = await CodingAgent(client, config).run(task, workspace)

        assert not result.finished
        assert result.stop_reason == "max_turns"
        assert result.turns == config.sandbox.max_turns

    @pytest.mark.asyncio
    async def test_plain_text_reply_ends_the_run(self, config, task):
        client = FakeClient([_response(text="I am done thinking.")])
        with Workspace.create() as workspace:
            result = await CodingAgent(client, config).run(task, workspace)

        assert result.stop_reason == "no_tool_call"
        assert not result.finished

    @pytest.mark.asyncio
    async def test_work_is_scored_even_without_finish(self, config, task):
        """An agent that runs out of turns still leaves files behind."""
        client = FakeClient(
            [_response(calls=[_call("write_file", path="word_freq.py", content="x=1")])] * 20
        )
        with Workspace.create() as workspace:
            await CodingAgent(client, config).run(task, workspace)
            assert workspace.list_files() == ["word_freq.py"]


class TestGuardrails:
    def _agent(self, config):
        return CodingAgent(FakeClient([]), config)

    def test_path_escape_is_recorded_and_not_raised(self, config, task):
        violations: list[str] = []
        with Workspace.create() as workspace:
            result = self._agent(config)._dispatch(
                "write_file",
                {"path": "../../evil.py", "content": "x"},
                workspace,
                task,
                violations,
            )

        assert "error" in str(result)
        assert any("escapes workspace" in v for v in violations)

    def test_writing_into_the_test_directory_is_recorded(self, config, task):
        violations: list[str] = []
        with Workspace.create() as workspace:
            self._agent(config)._dispatch(
                "write_file",
                {"path": "tests/test_free_pass.py", "content": "assert True"},
                workspace,
                task,
                violations,
            )

        assert any("test path" in v for v in violations)

    def test_reading_a_missing_file_returns_an_error_not_a_crash(self, config, task):
        with Workspace.create() as workspace:
            result = self._agent(config)._dispatch(
                "read_file", {"path": "nope.py"}, workspace, task, []
            )
        assert "no such file" in str(result)

    def test_unknown_tool_is_recorded(self, config, task):
        violations: list[str] = []
        with Workspace.create() as workspace:
            self._agent(config)._dispatch("rm_rf", {}, workspace, task, violations)
        assert any("unknown tool" in v for v in violations)

    def test_agent_cannot_read_the_test_suite(self, config, task):
        """Tests are not in the workspace at all, so there is nothing to read."""
        with Workspace.create() as workspace:
            result = self._agent(config)._dispatch(
                "read_file", {"path": "tests/test_word_freq.py"}, workspace, task, []
            )
        assert "no such file" in str(result)


class TestRunTests:
    @pytest.mark.asyncio
    async def test_held_out_results_are_never_shown_to_the_agent(self, config, task):
        from evolution.evaluator import Evaluator
        from evolution.sandbox import LocalBackend

        reference = (
            __import__("pathlib").Path(__file__).parent
            / "fixtures/word_freq/reference/word_freq.py"
        ).read_text(encoding="utf-8")

        agent = CodingAgent(FakeClient([]), config, evaluator=Evaluator(LocalBackend()))
        with Workspace.create() as workspace:
            workspace.write_file("word_freq.py", reference)
            result = agent._run_tests(workspace, task)

        assert result["passed"] == 8
        assert result["total"] == 8
        assert "held_out" not in result
