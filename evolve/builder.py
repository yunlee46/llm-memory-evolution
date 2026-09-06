"""Single-shot build: system = MD file verbatim, user = task spec. Returns the extracted artifact (html or python)."""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from .config import ROOT, load_config
from .llm import LLM

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_SNIFF = {"html": re.compile(r"<!doctype|<html", re.I), "python": re.compile(r"^\s*def predict\b", re.M)}


def extract_code(reply: str, lang: str = "html") -> str:
    """First fenced block that looks like the artifact for `lang`; else largest fenced block; else the whole reply."""
    reply = _THINK.sub("", reply)
    blocks = _FENCE.findall(reply)
    sniff = _SNIFF.get(lang)
    if sniff:
        for b in blocks:
            if sniff.search(b):
                return b.strip()
    if blocks:
        return max(blocks, key=len).strip()
    if lang == "html":
        m = re.search(r"<!doctype html.*", reply, re.I | re.S)
        return (m.group(0) if m else reply).strip()
    return reply.strip()


def extract_html(reply: str) -> str:
    return extract_code(reply, "html")


def load_spec(cfg: dict[str, Any]) -> str:
    return (ROOT / cfg["task"] / "spec.md").read_text()


def build_messages(md: str, spec: str) -> list[dict[str, str]]:
    msgs = []
    if md.strip():
        msgs.append({"role": "system", "content": md})
    msgs.append({"role": "user", "content": spec})
    return msgs


async def build(llm: LLM, cfg: dict[str, Any], md: str, spec: str) -> tuple[str, str, int]:
    """Returns (artifact_source, raw_reply, completion_tokens)."""
    mc = cfg["models"]
    reply, _, ct = await llm.chat_ex(mc["builder"], build_messages(md, spec),
                                     temperature=mc.get("builder_temperature"), max_tokens=mc.get("builder_max_tokens"))
    return extract_code(reply, cfg.get("artifact", {}).get("lang", "html")), reply, ct


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build once from an MD file and score the result")
    ap.add_argument("md", help="path to an instruction .md file")
    ap.add_argument("--out", default=None, help="where to write the artifact (default: runs/adhoc/<mdname>.<ext>)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    llm = LLM(cfg, mock=args.mock)
    md = Path(args.md).read_text()
    code, raw, ct = asyncio.run(build(llm, cfg, md, load_spec(cfg)))
    ext = Path(cfg["artifact"]["filename"]).suffix
    out = Path(args.out) if args.out else ROOT / "runs" / "adhoc" / (Path(args.md).stem + ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code)
    out.with_suffix(".reply.txt").write_text(raw)
    print(f"wrote {out} ({len(code)} chars); completion tokens {ct}")
    if not args.no_score:
        from .fitness import evaluate_artifact

        r = evaluate_artifact(out, cfg)
        print(f"score={r.score} {r.error}")
        for c in r.checks:
            print(f"  {'PASS' if c.passed else 'FAIL'} {c.name:<24} {c.detail}")


if __name__ == "__main__":
    main(sys.argv[1:])
