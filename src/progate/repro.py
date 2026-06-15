from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import platform
import random
import subprocess
import sys


def set_seed(seed: int) -> random.Random:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random.Random(seed)


def environment_snapshot(project_root: Path, argv: list[str]) -> dict[str, Any]:
    return {
        "argv": argv,
        "cwd": str(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git": {
            "commit": _run(["git", "rev-parse", "HEAD"], project_root),
            "status": _run(["git", "status", "--short"], project_root),
        },
        "uv": _run(["uv", "--version"], project_root),
    }


def _run(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    output = result.stdout.strip()
    if result.returncode != 0:
        return None
    return output
