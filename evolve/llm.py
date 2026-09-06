"""Thin OpenAI-compatible client for DeepInfra with retries, usage accounting and a mock mode."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ROOT, api_key, load_config


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, p: int, c: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += p
            self.completion_tokens += c
            m = self.by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            m["calls"] += 1
            m["prompt_tokens"] += p
            m["completion_tokens"] += c

    def to_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "by_model": self.by_model}


class LLM:
    def __init__(self, cfg: dict[str, Any], mock: bool = False):
        self.cfg = cfg["models"]
        self.mock = mock
        self.usage = Usage()
        self._sem = asyncio.Semaphore(self.cfg.get("concurrency", 8))
        if not mock:
            from openai import AsyncOpenAI, OpenAI

            self.client = AsyncOpenAI(api_key=api_key(), base_url=self.cfg["base_url"], max_retries=3)
            self.sync_client = OpenAI(api_key=api_key(), base_url=self.cfg["base_url"], max_retries=3)

    # -------------------------------------------------------------- mock
    def _mock_reply(self, model: str, messages: list[dict[str, str]]) -> str:
        if model == self.cfg["builder"]:
            return "```html\n" + (ROOT / "tasks/cat_bounce/reference/index.html").read_text() + "\n```"
        # mutator mock: echo the last markdown-looking chunk with a marker so lineage is visible
        import re

        text = messages[-1]["content"]
        m = re.search(r"---BEGIN---\n(.*?)\n---END---", text, re.S)
        body = m.group(1).strip() if m else text
        return f"---BEGIN---\n{body}\n\n<!-- mock-mutated {int(time.time() * 1000) % 100000} -->\n---END---"

    # -------------------------------------------------------------- calls
    async def chat(self, model: str, messages: list[dict[str, str]], temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        if self.mock:
            await asyncio.sleep(0.01)
            self.usage.add(model, 0, 0)
            return self._mock_reply(model, messages)
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        async with self._sem:
            for attempt in range(4):
                try:
                    resp = await self.client.chat.completions.create(**kwargs)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    wait = 2 ** attempt
                    print(f"[llm] {model}: {type(e).__name__}: {str(e)[:100]} — retry in {wait}s", file=sys.stderr)
                    await asyncio.sleep(wait)
        u = resp.usage
        self.usage.add(model, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
        return resp.choices[0].message.content or ""

    def chat_sync(self, *args, **kwargs) -> str:
        return asyncio.run(self.chat(*args, **kwargs))

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.sync_client.models.list().data)


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list model ids available on DeepInfra")
    ap.add_argument("--ping", action="store_true", help="one tiny call to each configured model")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    llm = LLM(cfg)
    if args.list:
        print("\n".join(llm.list_models()))
    if args.ping:
        for role in ("builder", "mutator"):
            m = cfg["models"][role]
            out = llm.chat_sync(m, [{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=20)
            print(f"{role} ({m}): {out.strip()[:60]}")
        print(json.dumps(llm.usage.to_dict(), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
