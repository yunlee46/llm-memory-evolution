"""Gemini API wrapper: retries, backoff, concurrency, token accounting.

Ten workers fire simultaneously each generation, so rate-limit handling is not
optional. Every call's usage_metadata is recorded — cost analysis depends on it.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from google import genai
from google.genai import errors, types

from .config import ApiConfig

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one or more calls."""

    prompt: int = 0
    output: int = 0
    thinking: int = 0
    cached: int = 0

    @property
    def total(self) -> int:
        # thoughts_token_count is reported separately but billed as output; the
        # API's own total_token_count is authoritative when present, so we only
        # use this derived total for aggregation across calls.
        return self.prompt + self.output + self.thinking

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt=self.prompt + other.prompt,
            output=self.output + other.output,
            thinking=self.thinking + other.thinking,
            cached=self.cached + other.cached,
        )

    @classmethod
    def from_response(cls, response: Any) -> TokenUsage:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return cls()
        return cls(
            prompt=usage.prompt_token_count or 0,
            output=usage.candidates_token_count or 0,
            thinking=usage.thoughts_token_count or 0,
            cached=usage.cached_content_token_count or 0,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt,
            "output_tokens": self.output,
            "thinking_tokens": self.thinking,
            "cached_tokens": self.cached,
            "total_tokens": self.total,
        }


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage
    model_id: str
    finish_reason: str | None = None
    function_calls: list[Any] = field(default_factory=list)
    raw: Any = None


class TokenLedger:
    """Running token total, keyed by role (worker / curator / judge)."""

    def __init__(self) -> None:
        self._by_role: dict[str, TokenUsage] = {}

    def record(self, role: str, usage: TokenUsage) -> None:
        self._by_role[role] = self._by_role.get(role, TokenUsage()) + usage

    def usage(self, role: str) -> TokenUsage:
        return self._by_role.get(role, TokenUsage())

    @property
    def total(self) -> TokenUsage:
        out = TokenUsage()
        for usage in self._by_role.values():
            out = out + usage
        return out

    def as_dict(self) -> dict[str, dict[str, int]]:
        data = {role: usage.as_dict() for role, usage in self._by_role.items()}
        data["total"] = self.total.as_dict()
        return data


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return None


def is_daily_quota_exhausted(exc: BaseException) -> bool:
    """True for a per-*day* quota 429, as opposed to a per-minute one.

    These look identical by status code but are operationally opposite: a
    per-minute limit clears in seconds, a daily one does not clear today. The
    quota id in the error details is what distinguishes them.
    """
    if _status_code(exc) != 429:
        return False
    haystack = " ".join(
        str(getattr(exc, attr, "")) for attr in ("details", "message", "response_json")
    ) + str(exc)
    return "PerDay" in haystack or "per day" in haystack.lower()


def is_retryable(exc: BaseException) -> bool:
    """Retry on rate limits, transient server errors, and connection failures."""
    if is_daily_quota_exhausted(exc):
        # Backing off 20s against a 24h quota just burns wall clock and then
        # fails anyway. Surface it immediately so the operator can react.
        return False
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return _status_code(exc) in RETRYABLE_STATUS
    if isinstance(exc, errors.APIError):
        return _status_code(exc) in RETRYABLE_STATUS
    return isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError))


class RequestBudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the configured request budget.

    On the free tier the binding constraint is requests per day, not tokens, so
    the harness counts requests and stops cleanly rather than discovering the
    limit as a wall of 429s partway through a generation.
    """


class GeminiClient:
    """Thin async wrapper over google-genai with the operational bits we need.

    Concurrency is bounded by a semaphore so that a generation's worth of
    workers doesn't stampede the rate limiter.
    """

    def __init__(
        self,
        api_key: str,
        api_config: ApiConfig,
        *,
        ledger: TokenLedger | None = None,
        request_budget: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._client = genai.Client(api_key=api_key)
        self._cfg = api_config
        self._semaphore = asyncio.Semaphore(api_config.concurrency)
        self.ledger = ledger or TokenLedger()
        self.request_budget = request_budget
        self.requests_by_model: dict[str, int] = {}

    @property
    def request_count(self) -> int:
        return sum(self.requests_by_model.values())

    def remaining_budget(self) -> int | None:
        if self.request_budget is None:
            return None
        return max(0, self.request_budget - self.request_count)

    def _charge_request(self, model: str) -> None:
        if self.request_budget is not None and self.request_count >= self.request_budget:
            raise RequestBudgetExceeded(
                f"request budget of {self.request_budget} exhausted "
                f"({self.requests_by_model})"
            )
        self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1

    # ---------------------------------------------------------------- models

    def list_models(self) -> list[dict[str, Any]]:
        """Model IDs this key can actually reach.

        Used by the smoke test to resolve model pinning empirically instead of
        guessing at ID strings (AGENTS.md §3).
        """
        out: list[dict[str, Any]] = []
        for model in self._client.models.list():
            actions = list(getattr(model, "supported_actions", None) or [])
            out.append(
                {
                    "name": model.name,
                    "display_name": getattr(model, "display_name", None),
                    "input_token_limit": getattr(model, "input_token_limit", None),
                    "output_token_limit": getattr(model, "output_token_limit", None),
                    "supported_actions": actions,
                }
            )
        return out

    def validate_model_ids(self, model_ids: Sequence[str]) -> list[str]:
        """Return the subset of model_ids the API does not expose."""
        available = set()
        for entry in self.list_models():
            name = entry["name"]
            available.add(name)
            available.add(name.removeprefix("models/"))
        return [m for m in model_ids if m not in available]

    # ------------------------------------------------------------ generation

    def _build_config(
        self,
        *,
        system_instruction: str | None,
        temperature: float,
        seed: int | None,
        thinking_budget: int | None,
        response_schema: Any | None,
        tools: Sequence[Any] | None,
        max_output_tokens: int | None,
    ) -> types.GenerateContentConfig:
        # Always off. We drive the tool loop ourselves (agent.py), and the SDK
        # auto-executing a model-requested function is not something an agent
        # harness should ever allow. Also silences the SDK's AFC warning.
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if seed is not None:
            kwargs["seed"] = seed
        if max_output_tokens:
            kwargs["max_output_tokens"] = max_output_tokens
        if thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        if response_schema is not None:
            # Structured output: the curator must return validated JSON, not
            # prose we have to parse (AGENTS.md §7).
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema
        if tools:
            kwargs["tools"] = list(tools)
        return types.GenerateContentConfig(**kwargs)

    async def generate(
        self,
        *,
        model: str,
        contents: Any,
        role: str = "worker",
        system_instruction: str | None = None,
        temperature: float = 0.7,
        seed: int | None = None,
        thinking_budget: int | None = None,
        response_schema: Any | None = None,
        tools: Sequence[Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """One generate_content call, with bounded concurrency and retries.

        Retries are safe here because generation is stateless; callers that own
        a sandbox must discard partial work themselves before retrying.
        """
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=temperature,
            seed=seed,
            thinking_budget=thinking_budget,
            response_schema=response_schema,
            tools=tools,
            max_output_tokens=max_output_tokens,
        )

        last_exc: BaseException | None = None
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                async with self._semaphore:
                    # Charged inside the retry loop: a retry is a real request
                    # against the quota, so it must count as one.
                    self._charge_request(model)
                    response = await self._client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    )
            except Exception as exc:  # noqa: BLE001 - re-raised below if fatal
                last_exc = exc
                if not is_retryable(exc) or attempt == self._cfg.max_retries:
                    raise
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Gemini call failed (attempt %d/%d, status=%s); retrying in %.1fs",
                    attempt,
                    self._cfg.max_retries,
                    _status_code(exc),
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            usage = TokenUsage.from_response(response)
            self.ledger.record(role, usage)
            return LLMResponse(
                text=response.text or "",
                usage=usage,
                model_id=model,
                finish_reason=self._finish_reason(response),
                function_calls=list(getattr(response, "function_calls", None) or []),
                raw=response,
            )

        assert last_exc is not None
        raise last_exc

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped by config."""
        ceiling = min(
            self._cfg.max_backoff_s,
            self._cfg.initial_backoff_s * (2 ** (attempt - 1)),
        )
        return random.uniform(self._cfg.initial_backoff_s, ceiling)

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        return str(reason) if reason is not None else None
