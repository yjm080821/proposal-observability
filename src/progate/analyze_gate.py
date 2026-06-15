from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import math
import statistics

from .analyze_pilot0 import rows_for_future_window
from .io import read_jsonl, write_json


def analyze_gate_behavior(
    run_dirs: list[Path],
    output_dir: Path,
    future_windows: list[int],
    low_alpha: float = 0.5,
    high_alpha: float = 0.8,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    active_step_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        raw_rows = read_jsonl(run_dir / "metrics.jsonl")
        seed = _seed(raw_rows, run_dir)
        active_rows = [row for row in raw_rows if row.get("T_e_active")]
        active_step_rows.extend(_active_step_rows(run_dir.name, seed, raw_rows))

        for window in future_windows:
            rows = rows_for_future_window(raw_rows, window)
            all_rows = _with_alpha(rows)
            active_window_rows = [row for row in all_rows if row.get("T_e_active")]
            for scope, scoped_rows in [("all_steps", all_rows), ("active_horizon", active_window_rows)]:
                bucket_rows.extend(_bucket_summary_rows(run_dir.name, seed, window, scope, scoped_rows, low_alpha, high_alpha))
                pair = _low_high_pair(run_dir.name, seed, window, scope, scoped_rows, low_alpha, high_alpha)
                if pair:
                    pair_rows.append(pair)
                correlation_rows.append(_correlation_row(run_dir.name, seed, window, scope, scoped_rows))

        if active_rows:
            bucket_rows.append(_immediate_spike_row(run_dir.name, seed, active_rows, low_alpha, high_alpha))

    aggregate_rows = _aggregate_pairs(pair_rows)
    _write_csv(output_dir / "gate_alpha_buckets.csv", bucket_rows)
    _write_csv(output_dir / "gate_low_high_pairs.csv", pair_rows)
    _write_csv(output_dir / "gate_pair_aggregate.csv", aggregate_rows)
    _write_csv(output_dir / "gate_correlations.csv", correlation_rows)
    _write_csv(output_dir / "gate_active_steps.csv", active_step_rows)
    write_json(
        output_dir / "gate_behavior_summary.json",
        {
            "run_ids": [path.name for path in run_dirs],
            "future_windows": future_windows,
            "low_alpha": low_alpha,
            "high_alpha": high_alpha,
            "direction_note": "future_val_delta = L_val(t+w) - L_val(t). Lower is better. Negative low-minus-high means low-alpha proposals were followed by better future movement; positive means high-alpha proposals were better.",
            "low_high_pairs": pair_rows,
            "low_high_aggregate": aggregate_rows,
        },
    )
    return output_dir


def _with_alpha(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("alpha_t") is not None and row.get("future_val_delta") is not None]


def _bucket_summary_rows(
    run_id: str,
    seed: int | str | None,
    window: int,
    scope: str,
    rows: list[dict[str, Any]],
    low_alpha: float,
    high_alpha: float,
) -> list[dict[str, Any]]:
    buckets = {
        "low": [row for row in rows if float(row["alpha_t"]) < low_alpha],
        "mid": [row for row in rows if low_alpha <= float(row["alpha_t"]) <= high_alpha],
        "high": [row for row in rows if float(row["alpha_t"]) > high_alpha],
    }
    output: list[dict[str, Any]] = []
    for bucket, bucket_rows in buckets.items():
        output.append(
            {
                "run_id": run_id,
                "seed": seed,
                "future_window": window,
                "scope": scope,
                "bucket": bucket,
                "count": len(bucket_rows),
                "alpha_mean": _mean([float(row["alpha_t"]) for row in bucket_rows]),
                "probe_score_mean": _mean([float(row["probe_score"]) for row in bucket_rows]),
                "future_val_delta_mean": _mean([float(row["future_val_delta"]) for row in bucket_rows]),
                "future_val_delta_sem": _sem([float(row["future_val_delta"]) for row in bucket_rows]),
                "loss_spike_fraction": _mean([1.0 if row.get("loss_spike") else 0.0 for row in bucket_rows]),
            }
        )
    return output


def _low_high_pair(
    run_id: str,
    seed: int | str | None,
    window: int,
    scope: str,
    rows: list[dict[str, Any]],
    low_alpha: float,
    high_alpha: float,
) -> dict[str, Any] | None:
    low_rows = [row for row in rows if float(row["alpha_t"]) < low_alpha]
    high_rows = [row for row in rows if float(row["alpha_t"]) > high_alpha]
    if not low_rows or not high_rows:
        return None
    low_delta = _mean([float(row["future_val_delta"]) for row in low_rows])
    high_delta = _mean([float(row["future_val_delta"]) for row in high_rows])
    low_probe = _mean([float(row["probe_score"]) for row in low_rows])
    high_probe = _mean([float(row["probe_score"]) for row in high_rows])
    return {
        "run_id": run_id,
        "seed": seed,
        "future_window": window,
        "scope": scope,
        "low_count": len(low_rows),
        "high_count": len(high_rows),
        "low_future_val_delta_mean": low_delta,
        "high_future_val_delta_mean": high_delta,
        "low_minus_high_future_val_delta": _subtract(low_delta, high_delta),
        "low_probe_score_mean": low_probe,
        "high_probe_score_mean": high_probe,
        "low_minus_high_probe_score": _subtract(low_probe, high_probe),
    }


def _correlation_row(run_id: str, seed: int | str | None, window: int, scope: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": seed,
        "future_window": window,
        "scope": scope,
        "count": len(rows),
        "alpha_future_val_delta_pearson": _pearson(
            [float(row["alpha_t"]) for row in rows],
            [float(row["future_val_delta"]) for row in rows],
        ),
        "probe_score_future_val_delta_pearson": _pearson(
            [float(row["probe_score"]) for row in rows],
            [float(row["future_val_delta"]) for row in rows],
        ),
        "alpha_probe_score_pearson": _pearson(
            [float(row["alpha_t"]) for row in rows],
            [float(row["probe_score"]) for row in rows],
        ),
    }


def _immediate_spike_row(
    run_id: str,
    seed: int | str | None,
    rows: list[dict[str, Any]],
    low_alpha: float,
    high_alpha: float,
) -> dict[str, Any]:
    low_rows = [row for row in rows if float(row["alpha_t"]) < low_alpha]
    high_rows = [row for row in rows if float(row["alpha_t"]) > high_alpha]
    return {
        "run_id": run_id,
        "seed": seed,
        "future_window": "immediate",
        "scope": "active_horizon",
        "bucket": "low_vs_high_spike",
        "count": len(rows),
        "alpha_mean": _mean([float(row["alpha_t"]) for row in rows]),
        "probe_score_mean": _mean([float(row["probe_score"]) for row in rows]),
        "future_val_delta_mean": None,
        "future_val_delta_sem": None,
        "loss_spike_fraction": _mean([1.0 if row.get("loss_spike") else 0.0 for row in rows]),
        "low_loss_spike_fraction": _mean([1.0 if row.get("loss_spike") else 0.0 for row in low_rows]),
        "high_loss_spike_fraction": _mean([1.0 if row.get("loss_spike") else 0.0 for row in high_rows]),
    }


def _active_step_rows(run_id: str, seed: int | str | None, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_step = {int(row["global_step"]): row for row in raw_rows if row.get("global_step") is not None}
    for row in raw_rows:
        if not row.get("T_e_active"):
            continue
        step = int(row["global_step"])
        next_row = by_step.get(step + 1)
        output.append(
            {
                "run_id": run_id,
                "seed": seed,
                "global_step": step,
                "alpha_t": row.get("alpha_t"),
                "alpha_probe": row.get("alpha_probe"),
                "probe_score": row.get("probe_score"),
                "train_loss": row.get("train_loss"),
                "validation_loss_snapshot": row.get("validation_loss_snapshot"),
                "next_validation_delta": _next_validation_delta(row, next_row),
                "loss_spike": row.get("loss_spike"),
            }
        )
    return output


def _aggregate_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row["future_window"], row["scope"]) for row in rows}, key=lambda item: (str(item[0]), str(item[1])))
    for window, scope in keys:
        group = [row for row in rows if row["future_window"] == window and row["scope"] == scope]
        gaps = [float(row["low_minus_high_future_val_delta"]) for row in group if row.get("low_minus_high_future_val_delta") is not None]
        probe_gaps = [float(row["low_minus_high_probe_score"]) for row in group if row.get("low_minus_high_probe_score") is not None]
        output.append(
            {
                "future_window": window,
                "scope": scope,
                "seed_count": len(group),
                "low_minus_high_future_val_delta_mean": _mean(gaps),
                "low_minus_high_future_val_delta_std": _std(gaps),
                "low_minus_high_probe_score_mean": _mean(probe_gaps),
                "low_minus_high_probe_score_std": _std(probe_gaps),
            }
        )
    return output


def _seed(rows: list[dict[str, Any]], path: Path) -> int | str | None:
    for row in rows:
        if row.get("seed") is not None:
            return row["seed"]
    marker = "_seed"
    if marker not in path.name:
        return None
    suffix = path.name.rsplit(marker, maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else suffix


def _next_validation_delta(row: dict[str, Any], next_row: dict[str, Any] | None) -> float | None:
    current = row.get("validation_loss_snapshot")
    future = next_row.get("validation_loss_snapshot") if next_row else None
    if current is None or future is None:
        return None
    return float(future) - float(current)


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    result = statistics.stdev(values)
    return None if math.isnan(result) else result


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
