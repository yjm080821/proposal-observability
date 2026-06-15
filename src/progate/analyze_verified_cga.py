from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import math
import statistics

from .analyze_pilot0 import rows_for_future_window
from .io import read_jsonl, write_json


def analyze_verified_cga(run_dirs: list[Path], output_dir: Path, future_windows: list[int]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        decision_rows.extend(_decision_rows(run_dir.name, metrics))
        for window in future_windows:
            future_rows.extend(_future_rows(run_dir.name, metrics, window))

    group_rows = _group_rows(future_rows)
    correlation_rows = _correlation_rows(future_rows)
    summary = {
        "direction_note": "Negative future_val_delta is better. Positive s2_minus_s1 means q2 probed better than q1.",
        "runs": [run_dir.name for run_dir in run_dirs],
        "future_windows": future_windows,
        "decision_counts": _counts(decision_rows, "decision_group"),
        "future_delta_by_group": group_rows,
        "s2_minus_s1_correlations": correlation_rows,
    }

    _write_csv(output_dir / "verified_cga_decision_groups.csv", decision_rows)
    _write_csv(output_dir / "verified_cga_future_delta_by_group.csv", group_rows)
    _write_csv(output_dir / "verified_cga_s2_minus_s1_correlation.csv", correlation_rows)
    write_json(output_dir / "verified_cga_summary.json", summary)
    return output_dir


def _decision_rows(run_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        output.append(
            {
                "run_id": run_id,
                "seed": row.get("seed"),
                "step": row.get("global_step", row.get("step")),
                "decision_group": _decision_group(row),
                "triggered": bool(row.get("cga_triggered")),
                "verified_commit": row.get("cga_verified_commit"),
                "commit_source": row.get("cga_commit_source"),
                "s1": row.get("cga_s1"),
                "s2": row.get("cga_s2"),
                "s2_minus_s1": row.get("cga_score_improvement"),
                "probe_score": row.get("probe_score"),
                "q1_norm": row.get("cga_q1_norm"),
                "q2_norm": row.get("cga_q2_norm"),
                "train_loss": row.get("train_loss"),
                "next_train_loss_delta": _next_train_loss_delta(row, next_row),
                "loss_spike": row.get("loss_spike"),
                "next_loss_spike": next_row.get("loss_spike") if next_row else None,
                "validation_loss_snapshot": row.get("validation_loss_snapshot"),
            }
        )
    return output


def _future_rows(run_id: str, rows: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows_for_future_window(rows, window):
        output.append(
            {
                "run_id": run_id,
                "seed": row.get("seed"),
                "step": row.get("global_step", row.get("step")),
                "future_window": window,
                "decision_group": _decision_group(row),
                "triggered": bool(row.get("cga_triggered")),
                "verified_commit": row.get("cga_verified_commit"),
                "commit_source": row.get("cga_commit_source"),
                "s1": row.get("cga_s1"),
                "s2": row.get("cga_s2"),
                "s2_minus_s1": row.get("cga_score_improvement"),
                "future_val_delta": row.get("future_val_delta"),
                "probe_score": row.get("probe_score"),
            }
        )
    return output


def _decision_group(row: dict[str, Any]) -> str:
    if row.get("cga_triggered") and row.get("cga_verified_commit") is True:
        return "triggered_q2_verified"
    if row.get("cga_triggered") and row.get("cga_verified_commit") is False:
        return "triggered_q1_kept"
    if row.get("cga_triggered"):
        return "triggered_q2_committed"
    return "non_triggered_q1"


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["future_window"], row["decision_group"]) for row in rows})
    output: list[dict[str, Any]] = []
    for window, group in keys:
        selected = [row for row in rows if row["future_window"] == window and row["decision_group"] == group]
        output.append(
            {
                "future_window": window,
                "decision_group": group,
                "count": len(selected),
                "future_val_delta_mean": _mean([row.get("future_val_delta") for row in selected]),
                "future_val_delta_sem": _sem([row.get("future_val_delta") for row in selected]),
                "s2_minus_s1_mean": _mean([row.get("s2_minus_s1") for row in selected]),
                "probe_score_mean": _mean([row.get("probe_score") for row in selected]),
            }
        )
    return output


def _correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    triggered = [row for row in rows if row.get("triggered") and row.get("s2_minus_s1") is not None]
    keys = sorted({row["future_window"] for row in triggered})
    output: list[dict[str, Any]] = []
    for window in keys:
        selected = [row for row in triggered if row["future_window"] == window]
        output.append(
            {
                "future_window": window,
                "scope": "triggered",
                "count": len(selected),
                "s2_minus_s1_future_val_delta_pearson": _pearson(
                    [row.get("s2_minus_s1") for row in selected],
                    [row.get("future_val_delta") for row in selected],
                ),
                "direction_note": "Negative suggests larger q2 score improvement is followed by lower future validation delta.",
            }
        )
    return output


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        output[value] = output.get(value, 0) + 1
    return dict(sorted(output.items()))


def _next_train_loss_delta(row: dict[str, Any], next_row: dict[str, Any] | None) -> float | None:
    if not next_row:
        return None
    current = row.get("train_loss")
    future = next_row.get("train_loss")
    if current is None or future is None:
        return None
    return future - current


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _clean(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _mean(values: list[float | None]) -> float | None:
    clean = _clean(values)
    return statistics.fmean(clean) if clean else None


def _sem(values: list[float | None]) -> float | None:
    clean = _clean(values)
    if len(clean) < 2:
        return None
    return statistics.stdev(clean) / math.sqrt(len(clean))


def _pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    clean_xs = [pair[0] for pair in pairs]
    clean_ys = [pair[1] for pair in pairs]
    mean_x = statistics.fmean(clean_xs)
    mean_y = statistics.fmean(clean_ys)
    var_x = sum((value - mean_x) ** 2 for value in clean_xs)
    var_y = sum((value - mean_y) ** 2 for value in clean_ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return covariance / math.sqrt(var_x * var_y)
