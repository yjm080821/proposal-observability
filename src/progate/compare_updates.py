from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import csv
import math
import statistics

from .io import read_jsonl, write_json


def compare_update_runs(
    baseline_dirs: list[Path],
    candidate_dirs: list[Path],
    output_dir: Path,
    early_steps: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = [_run_summary(path, "baseline", early_steps) for path in baseline_dirs]
    candidate_rows = [_run_summary(path, "candidate", early_steps) for path in candidate_dirs]

    run_rows = baseline_rows + candidate_rows
    pair_rows = _paired_rows(baseline_rows, candidate_rows)
    aggregate_rows = _aggregate_rows(run_rows, pair_rows)

    _write_csv(output_dir / "update_comparison_runs.csv", run_rows)
    _write_csv(output_dir / "update_comparison_pairs.csv", pair_rows)
    _write_csv(output_dir / "update_comparison_aggregate.csv", aggregate_rows)
    write_json(
        output_dir / "update_comparison_summary.json",
        {
            "early_steps": early_steps,
            "baseline_run_ids": [path.name for path in baseline_dirs],
            "candidate_run_ids": [path.name for path in candidate_dirs],
            "runs": run_rows,
            "pairs": pair_rows,
            "aggregate": aggregate_rows,
            "direction_note": "Lower validation loss/AUC and fewer loss spikes are better. Candidate deltas are candidate - baseline.",
        },
    )
    return output_dir


def _run_summary(path: Path, group: str, early_steps: int) -> dict[str, Any]:
    rows = read_jsonl(path / "metrics.jsonl")
    if not rows:
        raise ValueError(f"empty metrics file under {path}")

    validation_rows = [row for row in rows if row.get("validation_loss_snapshot") is not None]
    validation_losses = [float(row["validation_loss_snapshot"]) for row in validation_rows]
    train_losses = [float(row["train_loss"]) for row in rows if row.get("train_loss") is not None]
    early_validation = [
        float(row["validation_loss_snapshot"])
        for row in validation_rows
        if int(row["global_step"]) < early_steps
    ]
    first_20_steps = max(1, int(math.ceil(len(rows) * 0.2)))
    first_20_validation = [
        float(row["validation_loss_snapshot"])
        for row in validation_rows
        if int(row["global_step"]) < first_20_steps
    ]
    active_rows = [row for row in rows if row.get("T_e_active")]
    active_alphas = [float(row["alpha_t"]) for row in active_rows if row.get("alpha_t") is not None]
    active_q_norms = [float(row["q_t_norm"]) for row in active_rows if row.get("q_t_norm") is not None]
    active_effective_q_norms = [
        float(row["effective_q_norm"])
        for row in active_rows
        if row.get("effective_q_norm") is not None
    ]

    return {
        "group": group,
        "run_id": path.name,
        "seed": _seed(rows, path),
        "update_mode": rows[0].get("update_mode", "adamw"),
        "rows": len(rows),
        "early_steps": early_steps,
        "final_val_loss": _last(validation_losses),
        "best_val_loss": min(validation_losses) if validation_losses else None,
        "early_val_auc": _mean(early_validation),
        "first20_val_auc": _mean(first_20_validation),
        "mean_val_loss": _mean(validation_losses),
        "mean_train_loss": _mean(train_losses),
        "loss_spike_count": sum(1 for row in rows if row.get("loss_spike")),
        "max_train_loss_jump": _max_jump(train_losses),
        "active_horizon_steps": len(active_rows),
        "mean_alpha_in_horizon": _mean(active_alphas),
        "mean_effective_step_scale": _safe_ratio(sum(active_effective_q_norms), sum(active_q_norms)),
        "min_alpha_in_horizon": min(active_alphas) if active_alphas else None,
        "max_alpha_in_horizon": max(active_alphas) if active_alphas else None,
        "fraction_alpha_lt_0_5": _fraction(active_alphas, lambda value: value < 0.5),
        "fraction_alpha_gt_0_8": _fraction(active_alphas, lambda value: value > 0.8),
    }


def _paired_rows(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_seed = {row["seed"]: row for row in baseline_rows}
    candidate_by_seed = {row["seed"]: row for row in candidate_rows}
    output: list[dict[str, Any]] = []
    for seed in sorted(set(baseline_by_seed) & set(candidate_by_seed)):
        baseline = baseline_by_seed[seed]
        candidate = candidate_by_seed[seed]
        output.append(
            {
                "seed": seed,
                "baseline_run_id": baseline["run_id"],
                "candidate_run_id": candidate["run_id"],
                "delta_final_val_loss": _subtract(candidate["final_val_loss"], baseline["final_val_loss"]),
                "delta_best_val_loss": _subtract(candidate["best_val_loss"], baseline["best_val_loss"]),
                "delta_early_val_auc": _subtract(candidate["early_val_auc"], baseline["early_val_auc"]),
                "delta_first20_val_auc": _subtract(candidate["first20_val_auc"], baseline["first20_val_auc"]),
                "delta_loss_spike_count": _subtract(candidate["loss_spike_count"], baseline["loss_spike_count"]),
                "delta_max_train_loss_jump": _subtract(candidate["max_train_loss_jump"], baseline["max_train_loss_jump"]),
                "candidate_mean_alpha_in_horizon": candidate["mean_alpha_in_horizon"],
                "candidate_mean_effective_step_scale": candidate["mean_effective_step_scale"],
                "candidate_fraction_alpha_lt_0_5": candidate["fraction_alpha_lt_0_5"],
                "candidate_fraction_alpha_gt_0_8": candidate["fraction_alpha_gt_0_8"],
            }
        )
    return output


def _aggregate_rows(run_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in sorted({row["group"] for row in run_rows}):
        rows = [row for row in run_rows if row["group"] == group]
        output.extend(_metric_aggregate_rows(group, rows))
    output.extend(_metric_aggregate_rows("paired_delta", pair_rows))
    return output


def _metric_aggregate_rows(scope: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "final_val_loss",
        "best_val_loss",
        "early_val_auc",
        "first20_val_auc",
        "loss_spike_count",
        "max_train_loss_jump",
        "mean_alpha_in_horizon",
        "mean_effective_step_scale",
        "fraction_alpha_lt_0_5",
        "fraction_alpha_gt_0_8",
        "delta_final_val_loss",
        "delta_best_val_loss",
        "delta_early_val_auc",
        "delta_first20_val_auc",
        "delta_loss_spike_count",
        "delta_max_train_loss_jump",
        "candidate_mean_alpha_in_horizon",
        "candidate_mean_effective_step_scale",
        "candidate_fraction_alpha_lt_0_5",
        "candidate_fraction_alpha_gt_0_8",
    ]
    output: list[dict[str, Any]] = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not values:
            continue
        output.append(
            {
                "scope": scope,
                "metric": metric,
                "n": len(values),
                "mean": _mean(values),
                "std": _std(values),
                "min": min(values),
                "max": max(values),
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


def _last(values: list[float]) -> float | None:
    return values[-1] if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    result = statistics.stdev(values)
    return None if math.isnan(result) else result


def _fraction(values: list[float], predicate: Callable[[float], bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if predicate(value)) / len(values)


def _max_jump(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return max(values[index] - values[index - 1] for index in range(1, len(values)))


def _subtract(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


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
