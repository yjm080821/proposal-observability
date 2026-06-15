from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re


def create_run_dir(root: Path, name: str, seed: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(name)
    run_dir = root / f"{stamp}_{slug}_seed{seed}"
    index = 1
    candidate = run_dir
    while candidate.exists():
        index += 1
        candidate = root / f"{run_dir.name}_{index}"
    candidate.mkdir(parents=True)
    return candidate


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    return slug.strip("-") or "run"
