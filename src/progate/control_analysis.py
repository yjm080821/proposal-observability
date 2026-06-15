from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math
import statistics

from .io import read_jsonl, write_json


def analyze_control_runs(
    run_dirs: list[Path],
    output_dir: Path,
    bad_run_threshold: float | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [_summarize_run(path) for path in run_dirs]
    threshold = bad_run_threshold
    if threshold is None:
        threshold = _adamw_worst20_threshold(run_rows)
    aggregate_rows = _aggregate_methods(run_rows, threshold)
    paired_rows = _paired_seed_rows(run_rows)

    _write_csv(output_dir / "control_run_summary.csv", run_rows)
    _write_csv(output_dir / "control_method_summary.csv", aggregate_rows)
    _write_csv(output_dir / "control_paired_seed_summary.csv", paired_rows)
    write_json(
        output_dir / "control_analysis_summary.json",
        {
            "run_ids": [path.name for path in run_dirs],
            "bad_run_threshold": threshold,
            "bad_run_threshold_note": (
                "If not provided, the threshold is the AdamW final-validation "
                "worst-20% quantile within this analysis set."
            ),
            "runs": run_rows,
            "aggregate": aggregate_rows,
        },
    )
    return output_dir


def _summarize_run(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path / "metrics.jsonl")
    if not rows:
        raise ValueError(f"empty metrics under {path}")
    validation_rows = [row for row in rows if row.get("validation_loss_snapshot") is not None]
    validation_losses = [float(row["validation_loss_snapshot"]) for row in validation_rows]
    final_val = validation_losses[-1] if validation_losses else None
    method = _method_name(rows, path)
    total_backward, extra_backward = _backward_counts(rows)
    return {
        "run_id": path.name,
        "method": method,
        "seed": _seed(rows, path),
        "model_name": rows[0].get("model_name"),
        "dataset_name": rows[0].get("dataset_name"),
        "steps": len(rows),
        "validation_snapshots": len(validation_rows),
        "final_val_loss": final_val,
        "best_val_loss": min(validation_losses) if validation_losses else None,
        "mean_val_loss": statistics.fmean(validation_losses) if validation_losses else None,
        "val_auc_step_axis": _normalized_auc(
            [(int(row["global_step"]), float(row["validation_loss_snapshot"])) for row in validation_rows]
        ),
        "val_auc_backward_axis": _normalized_auc(
            [
                (x_value, float(row["validation_loss_snapshot"]))
                for x_value, row in _validation_rows_with_backward_axis(rows)
            ]
        ),
        "total_backward_count": total_backward,
        "extra_backward_count": extra_backward,
        "extra_backward_rate": extra_backward / len(rows) if rows else None,
        "control_trigger_rate": _mean([row.get("control_triggered") for row in rows]),
        "mean_commit_alpha": _mean([row.get("commit_alpha") for row in rows]),
        "mean_score_improvement_on_triggered": _mean(
            [row.get("score_improvement") for row in rows if row.get("control_triggered")]
        ),
        "fraction_score_improved_on_triggered": _fraction(
            [row.get("score_improvement") for row in rows if row.get("control_triggered")],
            lambda value: value > 0.0,
        ),
        "loss_spike_count": sum(1 for row in rows if row.get("loss_spike")),
    }


def _method_name(rows: list[dict[str, Any]], path: Path) -> str:
    control_mode = rows[0].get("control_mode", "off")
    if control_mode and control_mode != "off":
        return str(control_mode)
    run_id = path.name
    if "adamw-longer" in run_id or "adamw_longer" in run_id:
        return "adamw_longer_1p25"
    return str(rows[0].get("update_mode", "adamw"))


def _backward_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    if rows[-1].get("cumulative_backward_count") is not None:
        return (
            int(rows[-1].get("cumulative_backward_count") or len(rows)),
            int(rows[-1].get("cumulative_extra_backward_count") or 0),
        )
    extra = sum(_row_extra_backward_count(row) for row in rows)
    return len(rows) + extra, extra


def _row_extra_backward_count(row: dict[str, Any]) -> int:
    if row.get("backward_count_extra") is not None:
        return int(row.get("backward_count_extra") or 0)
    return int(row.get("extra_backward_count") or 0)


def _validation_rows_with_backward_axis(rows: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    output: list[tuple[int, dict[str, Any]]] = []
    cumulative = 0
    for row in rows:
        if row.get("cumulative_backward_count") is not None:
            cumulative = int(row["cumulative_backward_count"])
        else:
            cumulative += 1 + _row_extra_backward_count(row)
        if row.get("validation_loss_snapshot") is not None:
            output.append((cumulative, row))
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


def _adamw_worst20_threshold(rows: list[dict[str, Any]]) -> float | None:
    values = sorted(
        float(row["final_val_loss"])
        for row in rows
        if row.get("method") == "adamw" and row.get("final_val_loss") is not None
    )
    if not values:
        return None
    return _percentile(values, 0.8)


def _aggregate_methods(rows: list[dict[str, Any]], bad_run_threshold: float | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        finals = [float(row["final_val_loss"]) for row in group if row.get("final_val_loss") is not None]
        output.append(
            {
                "method": method,
                "n": len(group),
                "mean_final_val_loss": _mean(finals),
                "worst_final_val_loss": max(finals) if finals else None,
                "best_final_val_loss": min(finals) if finals else None,
                "std_final_val_loss": _std(finals),
                "bad_run_threshold": bad_run_threshold,
                "bad_run_rate": _bad_run_rate(finals, bad_run_threshold),
                "mean_val_auc_step_axis": _mean([row.get("val_auc_step_axis") for row in group]),
                "mean_val_auc_backward_axis": _mean([row.get("val_auc_backward_axis") for row in group]),
                "mean_total_backward_count": _mean([row.get("total_backward_count") for row in group]),
                "mean_extra_backward_rate": _mean([row.get("extra_backward_rate") for row in group]),
                "mean_control_trigger_rate": _mean([row.get("control_trigger_rate") for row in group]),
                "mean_score_improvement_on_triggered": _mean(
                    [row.get("mean_score_improvement_on_triggered") for row in group]
                ),
                "fraction_score_improved_on_triggered": _mean(
                    [row.get("fraction_score_improved_on_triggered") for row in group]
                ),
            }
        )
    return output


def _paired_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seeds = sorted({row["seed"] for row in rows if row.get("seed") is not None})
    methods = sorted({str(row["method"]) for row in rows})
    output: list[dict[str, Any]] = []
    for seed in seeds:
        row: dict[str, Any] = {"seed": seed}
        for method in methods:
            matches = [item for item in rows if item.get("seed") == seed and item.get("method") == method]
            if matches:
                row[f"{method}_final_val_loss"] = matches[0].get("final_val_loss")
                row[f"{method}_val_auc_backward_axis"] = matches[0].get("val_auc_backward_axis")
        output.append(row)
    return output


def _normalized_auc(points: list[tuple[int, float]]) -> float | None:
    if not points:
        return None
    ordered = sorted(points)
    if len(ordered) == 1:
        return ordered[0][1]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    span = ordered[-1][0] - ordered[0][0]
    if span <= 0:
        return statistics.fmean(value for _, value in ordered)
    return area / span


def _bad_run_rate(values: list[float], threshold: float | None) -> float | None:
    if threshold is None or not values:
        return None
    return sum(1 for value in values if value >= threshold) / len(values)


def _mean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None and not isinstance(value, bool)]
    bools = [float(value) for value in values if isinstance(value, bool)]
    clean.extend(bools)
    return statistics.fmean(clean) if clean else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    result = statistics.stdev(values)
    return None if math.isnan(result) else result


def _fraction(values: list[Any], predicate: Any) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(1 for value in clean if predicate(value)) / len(clean)


def _percentile(sorted_values: list[float], probability: float) -> float:
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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
