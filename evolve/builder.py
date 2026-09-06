"""Single-shot build: system = MD file verbatim, user = task spec. Returns extracted index.html."""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from .config import ROOT, load_config
from .llm import LLM

_HTML_FENCE = re.compile(r"```(?:html|HTML)?[^\n]*\n(.*?)```", re.S)


def extract_html(reply: str) -> str:
    """First fenced block containing <html> or <!doctype>; else largest fenced block; else the whole reply."""
    blocks = _HTML_FENCE.findall(reply)
    for b in blocks:
        if re.search(r"<!doctype|<html", b, re.I):
            return b.strip()
    if blocks:
        return max(blocks, key=len).strip()
    m = re.search(r"<!doctype html.*", reply, re.I | re.S)
    return (m.group(0) if m else reply).strip()


def load_spec(cfg: dict[str, Any]) -> str:
    return (ROOT / cfg["task"] / "spec.md").read_text()


def build_messages(md: str, spec: str) -> list[dict[str, str]]:
    msgs = []
    if md.strip():
        msgs.append({"role": "system", "content": md})
    msgs.append({"role": "user", "content": spec})
    return msgs


async def build(llm: LLM, cfg: dict[str, Any], md: str, spec: str) -> tuple[str, str]:
    """Returns (html, raw_reply)."""
    mc = cfg["models"]
    reply = await llm.chat(mc["builder"], build_messages(md, spec),
                           temperature=mc.get("builder_temperature"), max_tokens=mc.get("builder_max_tokens"))
    return extract_html(reply), reply


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build once from an MD file and score the result")
    ap.add_argument("md", help="path to an instruction .md file")
    ap.add_argument("--out", default=None, help="where to write index.html (default: runs/adhoc/<mdname>.html)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    llm = LLM(cfg, mock=args.mock)
    md = Path(args.md).read_text()
    html, raw = asyncio.run(build(llm, cfg, md, load_spec(cfg)))
    out = Path(args.out) if args.out else ROOT / "runs" / "adhoc" / (Path(args.md).stem + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    out.with_suffix(".reply.txt").write_text(raw)
    print(f"wrote {out} ({len(html)} chars); usage {llm.usage.to_dict()}")
    if not args.no_score:
        from .fitness import evaluate_html

        r = evaluate_html(out, cfg)
        print(f"score={r.score} {r.error}")
        for c in r.checks:
            print(f"  {'PASS' if c.passed else 'FAIL'} {c.name:<24} {c.detail}")


if __name__ == "__main__":
    main(sys.argv[1:])
