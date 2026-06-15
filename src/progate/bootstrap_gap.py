from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math
import random
import statistics

from .analyze_pilot0 import rows_for_future_window
from .io import read_jsonl, write_json


def bootstrap_top_bottom_gap(
    run_dir: Path,
    future_windows: list[int],
    top_fraction: float = 0.2,
    bootstrap_n: int = 5000,
    seed: int = 20260511,
) -> Path:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _resolve_run_dirs(run_dir)
    per_seed_rows: list[dict[str, Any]] = []
    pooled_by_window: dict[int, list[_WindowGroups]] = {window: [] for window in future_windows}

    rng = random.Random(seed)
    for candidate in run_dirs:
        raw_rows = read_jsonl(candidate / "metrics.jsonl")
        run_seed = _run_seed(raw_rows, candidate)
        for window in future_windows:
            window_rows = rows_for_future_window(raw_rows, window)
            groups = _top_bottom_groups(window_rows, top_fraction)
            if groups is None:
                continue
            pooled_by_window[window].append(groups)
            result = _bootstrap_group_gap(groups.top, groups.bottom, bootstrap_n, rng)
            per_seed_rows.append(
                {
                    "scope": "per_seed",
                    "seed": run_seed,
                    "run_id": candidate.name,
                    "future_window": window,
                    "gap_mean": result["gap_mean"],
                    "gap_std": result["gap_std"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "n_top": len(groups.top),
                    "n_bottom": len(groups.bottom),
                    "bootstrap_n": bootstrap_n,
                    "top_fraction": top_fraction,
                }
            )

    aggregate_rows = _seed_aggregate_rows(per_seed_rows)
    pooled_rows = _pooled_rows(pooled_by_window, bootstrap_n, rng, top_fraction)
    output_rows = per_seed_rows + aggregate_rows + pooled_rows

    _write_csv(analysis_dir / "bootstrap_top_bottom_gap.csv", output_rows)
    write_json(
        analysis_dir / "bootstrap_top_bottom_gap_summary.json",
        {
            "input_dir": str(run_dir),
            "run_ids": [path.name for path in run_dirs],
            "future_windows": future_windows,
            "top_fraction": top_fraction,
            "bootstrap_n": bootstrap_n,
            "bootstrap_seed": seed,
            "direction_note": "future_val_delta = L_val(t+w) - L_val(t). Lower is better, so negative top-minus-bottom gap supports ProGate.",
            "per_seed": per_seed_rows,
            "seed_aggregate": aggregate_rows,
            "pooled_stratified": pooled_rows,
        },
    )
    return analysis_dir


class _WindowGroups:
    def __init__(self, top: list[float], bottom: list[float]) -> None:
        self.top = top
        self.bottom = bottom


def _resolve_run_dirs(path: Path) -> list[Path]:
    if (path / "metrics.jsonl").exists():
        return [path]

    summary_path = path / "sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"expected metrics.jsonl or sweep_summary.json under {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output: list[Path] = []
    for row in summary.get("runs", []):
        run_id = row.get("run_id")
        if not run_id:
            continue
        candidate = Path(run_id)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if not (candidate / "metrics.jsonl").exists():
            raise FileNotFoundError(f"missing metrics.jsonl for {candidate}")
        output.append(candidate)
    return output


def _run_seed(rows: list[dict[str, Any]], run_dir: Path) -> int | str | None:
    for row in rows:
        if row.get("seed") is not None:
            return row["seed"]
    return _seed_from_name(run_dir.name)


def _seed_from_name(value: str) -> int | str | None:
    marker = "_seed"
    if marker not in value:
        return None
    suffix = value.rsplit(marker, maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else suffix


def _top_bottom_groups(rows: list[dict[str, Any]], fraction: float) -> _WindowGroups | None:
    rows = [row for row in rows if row.get("probe_score") is not None and row.get("future_val_delta") is not None]
    if not rows:
        return None
    count = max(1, int(round(len(rows) * fraction)))
    ordered = sorted(rows, key=lambda row: float(row["probe_score"]))
    bottom = [float(row["future_val_delta"]) for row in ordered[:count]]
    top = [float(row["future_val_delta"]) for row in ordered[-count:]]
    return _WindowGroups(top=top, bottom=bottom)


def _bootstrap_group_gap(
    top: list[float],
    bottom: list[float],
    bootstrap_n: int,
    rng: random.Random,
) -> dict[str, float | None]:
    observed = _gap(top, bottom)
    samples: list[float] = []
    for _ in range(bootstrap_n):
        top_sample = [top[rng.randrange(len(top))] for _ in top]
        bottom_sample = [bottom[rng.randrange(len(bottom))] for _ in bottom]
        samples.append(_gap(top_sample, bottom_sample))
    samples.sort()
    return {
        "gap_mean": observed,
        "gap_std": _std(samples),
        "ci_low": _percentile(samples, 0.025),
        "ci_high": _percentile(samples, 0.975),
    }


def _seed_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    windows = sorted({int(row["future_window"]) for row in rows})
    for window in windows:
        group = [row for row in rows if int(row["future_window"]) == window]
        gaps = [float(row["gap_mean"]) for row in group if row.get("gap_mean") is not None]
        output.append(
            {
                "scope": "seed_aggregate",
                "seed": None,
                "run_id": None,
                "future_window": window,
                "gap_mean": _mean(gaps),
                "gap_std": _std(gaps),
                "ci_low": None,
                "ci_high": None,
                "n_top": sum(int(row["n_top"]) for row in group),
                "n_bottom": sum(int(row["n_bottom"]) for row in group),
                "bootstrap_n": None,
                "top_fraction": group[0]["top_fraction"] if group else None,
                "seed_count": len(group),
            }
        )
    return output


def _pooled_rows(
    pooled_by_window: dict[int, list[_WindowGroups]],
    bootstrap_n: int,
    rng: random.Random,
    top_fraction: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for window, groups in sorted(pooled_by_window.items()):
        if not groups:
            continue
        top = [value for group in groups for value in group.top]
        bottom = [value for group in groups for value in group.bottom]
        observed = _gap(top, bottom)
        samples: list[float] = []
        for _ in range(bootstrap_n):
            top_sample: list[float] = []
            bottom_sample: list[float] = []
            for group in groups:
                top_sample.extend(group.top[rng.randrange(len(group.top))] for _ in group.top)
                bottom_sample.extend(group.bottom[rng.randrange(len(group.bottom))] for _ in group.bottom)
            samples.append(_gap(top_sample, bottom_sample))
        samples.sort()
        output.append(
            {
                "scope": "pooled_stratified",
                "seed": None,
                "run_id": None,
                "future_window": window,
                "gap_mean": observed,
                "gap_std": _std(samples),
                "ci_low": _percentile(samples, 0.025),
                "ci_high": _percentile(samples, 0.975),
                "n_top": len(top),
                "n_bottom": len(bottom),
                "bootstrap_n": bootstrap_n,
                "top_fraction": top_fraction,
                "seed_count": len(groups),
            }
        )
    return output


def _gap(top: list[float], bottom: list[float]) -> float:
    return statistics.fmean(top) - statistics.fmean(bottom)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    result = statistics.stdev(values)
    return None if math.isnan(result) else result


def _percentile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
