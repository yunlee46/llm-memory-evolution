from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(p)
    meta_path = ROOT / cfg.get("task", "") / "task.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    cfg["task_meta"] = meta or {}
    cfg.setdefault("artifact", cfg["task_meta"].get("artifact") or {"filename": "index.html", "lang": "html"})
    return cfg


def api_key() -> str:
    key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not key:
        raise SystemExit("DEEPINFRA_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key
