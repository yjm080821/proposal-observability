from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math
import statistics

from .io import read_jsonl, write_json


def analyze_run(
    run_dir: Path,
    bins: int = 5,
    alpha_acc: float = 0.8,
    alpha_rej: float = 0.1,
    top_fraction: float = 0.2,
    future_windows: list[int] | None = None,
) -> Path:
    metrics_path = run_dir / "metrics.jsonl"
    raw_rows = read_jsonl(metrics_path)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logged_rows = [row for row in raw_rows if row.get("future_val_delta") is not None]
    window_outputs: dict[str, Any] = {}

    if logged_rows:
        _write_analysis_set(analysis_dir, logged_rows, bins, alpha_acc, alpha_rej, top_fraction)
        window_outputs["logged"] = correlation_summary(logged_rows)

    windows = future_windows or [1, 5, 10, 20]
    first_window_rows: list[dict[str, Any]] | None = None
    for window in windows:
        window_rows = rows_for_future_window(raw_rows, window)
        if not window_rows:
            continue
        if first_window_rows is None:
            first_window_rows = window_rows
        window_dir = analysis_dir / f"w{window}"
        window_dir.mkdir(parents=True, exist_ok=True)
        _write_analysis_set(window_dir, window_rows, bins, alpha_acc, alpha_rej, top_fraction)
        window_outputs[f"w{window}"] = correlation_summary(window_rows)

    if not logged_rows and first_window_rows:
        _write_analysis_set(analysis_dir, first_window_rows, bins, alpha_acc, alpha_rej, top_fraction)

    write_json(analysis_dir / "future_windows_summary.json", window_outputs)
    return analysis_dir


def rows_for_future_window(rows: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    if window <= 0:
        return []
    by_step = {
        int(row["global_step"]): row
        for row in rows
        if row.get("validation_loss_snapshot") is not None and row.get("global_step") is not None
    }
    output: list[dict[str, Any]] = []
    for step in sorted(by_step):
        future = by_step.get(step + window)
        if future is None:
            continue
        row = dict(by_step[step])
        row["future_window_w"] = window
        row["future_window"] = window
        row["future_val_delta"] = future["validation_loss_snapshot"] - row["validation_loss_snapshot"]
        output.append(row)
    return output


def _write_analysis_set(
    output_dir: Path,
    rows: list[dict[str, Any]],
    bins: int,
    alpha_acc: float,
    alpha_rej: float,
    top_fraction: float,
) -> None:
    score_bins = score_bin_rows(rows, bins)
    threshold_summary = accepted_rejected_summary(rows, alpha_acc, alpha_rej)
    percentile_summary = percentile_summary_rows(rows, top_fraction)
    correlation = correlation_summary(rows)

    _write_csv(output_dir / "probe_score_bins.csv", score_bins)
    write_json(
        output_dir / "accepted_rejected_summary.json",
        {
            "alpha_threshold": threshold_summary,
            "percentile": percentile_summary,
        },
    )
    write_json(output_dir / "correlation.json", correlation)


def score_bin_rows(rows: list[dict[str, Any]], bins: int) -> list[dict[str, Any]]:
    if not rows or bins <= 0:
        return []
    ordered = sorted(rows, key=lambda row: row["probe_score"])
    output: list[dict[str, Any]] = []
    for index in range(bins):
        start = index * len(ordered) // bins
        end = (index + 1) * len(ordered) // bins
        chunk = ordered[start:end]
        deltas = [row["future_val_delta"] for row in chunk]
        scores = [row["probe_score"] for row in chunk]
        output.append(
            {
                "bin": index,
                "count": len(chunk),
                "probe_score_min": _first(scores),
                "probe_score_max": _last(scores),
                "probe_score_mean": _mean(scores),
                "future_val_delta_mean": _mean(deltas),
                "future_val_delta_sem": _sem(deltas),
            }
        )
    return output


def accepted_rejected_summary(rows: list[dict[str, Any]], alpha_acc: float, alpha_rej: float) -> dict[str, Any]:
    accepted = [row for row in rows if row["alpha_probe"] >= alpha_acc]
    rejected = [row for row in rows if row["alpha_probe"] <= alpha_rej]
    middle = [row for row in rows if alpha_rej < row["alpha_probe"] < alpha_acc]
    return {
        "alpha_acc": alpha_acc,
        "alpha_rej": alpha_rej,
        "accepted": _group_summary(accepted),
        "rejected": _group_summary(rejected),
        "middle": _group_summary(middle),
    }


def percentile_summary_rows(rows: list[dict[str, Any]], fraction: float) -> dict[str, Any]:
    if not rows:
        return {"fraction": fraction, "top": _group_summary([]), "bottom": _group_summary([])}
    count = max(1, int(round(len(rows) * fraction)))
    ordered = sorted(rows, key=lambda row: row["probe_score"])
    return {
        "fraction": fraction,
        "bottom": _group_summary(ordered[:count]),
        "top": _group_summary(ordered[-count:]),
    }


def correlation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["probe_score"] for row in rows]
    deltas = [row["future_val_delta"] for row in rows]
    return {
        "n": len(rows),
        "probe_score_future_val_delta_pearson": _pearson(scores, deltas),
        "direction_note": "Negative is better when future_val_delta = L_val(t+w) - L_val(t).",
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["future_val_delta"] for row in rows]
    scores = [row["probe_score"] for row in rows]
    return {
        "count": len(rows),
        "probe_score_mean": _mean(scores),
        "future_val_delta_mean": _mean(deltas),
        "future_val_delta_sem": _sem(deltas),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _first(values: list[float]) -> float | None:
    return values[0] if values else None


def _last(values: list[float]) -> float | None:
    return values[-1] if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sem(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(var_x * var_y)
