from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import statistics

from .io import write_json


def analyze_cga_diagnostic(run_dirs: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summarize_run(run_dir) for run_dir in run_dirs]
    _write_csv(output_dir / "cga_diagnostic_runs.csv", rows)
    write_json(
        output_dir / "cga_diagnostic_summary.json",
        {
            "runs": rows,
            "aggregate": _aggregate(rows),
            "direction_note": "Positive score_improvement means q2 probe score is better than q1 on CGA-triggered low-score steps.",
        },
    )
    return output_dir


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    metrics = [_parse_json_line(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not metrics:
        raise ValueError(f"no metrics rows in {run_dir}")

    triggered = [row for row in metrics if row.get("cga_triggered")]
    ga2_rows = [row for row in metrics if row.get("ga2_recorded")]
    improvements = [row["cga_score_improvement"] for row in triggered if row.get("cga_score_improvement") is not None]
    ga2_improvements = [row["ga2_score_improvement"] for row in ga2_rows if row.get("ga2_score_improvement") is not None]
    s1_values = [row["probe_score"] for row in metrics if row.get("probe_score") is not None]
    cga_policy_scores = [
        row["cga_s2"] if row.get("cga_triggered") and row.get("cga_s2") is not None else row["probe_score"]
        for row in metrics
        if row.get("probe_score") is not None
    ]
    triggered_s1 = [row["cga_s1"] for row in triggered if row.get("cga_s1") is not None]
    triggered_s2 = [row["cga_s2"] for row in triggered if row.get("cga_s2") is not None]
    triggered_ga2 = [row["ga2_score"] for row in triggered if row.get("ga2_score") is not None]
    mean_s1_all = _mean(s1_values)
    mean_cga_policy_score_all = _mean(cga_policy_scores)
    mean_ga2_score_all = _mean([row["ga2_score"] for row in ga2_rows if row.get("ga2_score") is not None])
    return {
        "run_id": run_dir.name,
        "seed": metrics[0]["seed"],
        "rows": summary["rows"],
        "triggered_steps": summary.get("cga_triggered_steps", len(triggered)),
        "trigger_rate": summary.get("cga_trigger_rate", _safe_div(len(triggered), len(metrics))),
        "extra_backward_overhead": summary.get(
            "cga_extra_backward_overhead",
            _safe_div(sum(int(row.get("extra_backward_count") or 0) for row in metrics), len(metrics)),
        ),
        "mean_score_improvement": _mean(improvements),
        "median_score_improvement": _median(improvements),
        "fraction_score_improved": _fraction(improvements, lambda value: value > 0.0),
        "min_score_improvement": min(improvements) if improvements else None,
        "max_score_improvement": max(improvements) if improvements else None,
        "ga2_recorded_steps": len(ga2_rows),
        "ga2_extra_backward_overhead": _safe_div(
            sum(int(row.get("ga2_extra_backward_count") or 0) for row in metrics),
            len(metrics),
        ),
        "ga2_mean_score_improvement_all": _mean(ga2_improvements),
        "ga2_median_score_improvement_all": _median(ga2_improvements),
        "ga2_fraction_score_improved_all": _fraction(ga2_improvements, lambda value: value > 0.0),
        "mean_s1_all": mean_s1_all,
        "mean_cga_policy_score_all": mean_cga_policy_score_all,
        "mean_ga2_score_all": mean_ga2_score_all,
        "cga_policy_score_improvement_all": _nullable_sub(mean_cga_policy_score_all, mean_s1_all),
        "ga2_score_improvement_all_from_means": _nullable_sub(mean_ga2_score_all, mean_s1_all),
        "cga_policy_minus_ga2_score_all": _nullable_sub(mean_cga_policy_score_all, mean_ga2_score_all),
        "mean_triggered_s1": _mean(triggered_s1),
        "mean_triggered_s2": _mean(triggered_s2),
        "mean_triggered_ga2": _mean(triggered_ga2),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for key in [
        "trigger_rate",
        "extra_backward_overhead",
        "mean_score_improvement",
        "median_score_improvement",
        "fraction_score_improved",
        "ga2_extra_backward_overhead",
        "ga2_mean_score_improvement_all",
        "ga2_median_score_improvement_all",
        "ga2_fraction_score_improved_all",
        "mean_s1_all",
        "mean_cga_policy_score_all",
        "mean_ga2_score_all",
        "cga_policy_score_improvement_all",
        "ga2_score_improvement_all_from_means",
        "cga_policy_minus_ga2_score_all",
        "mean_triggered_s1",
        "mean_triggered_s2",
        "mean_triggered_ga2",
    ]:
        values = [row[key] for row in rows if row.get(key) is not None]
        aggregate[key] = _stats(values)
    aggregate["triggered_steps"] = {
        "mean": _mean([row["triggered_steps"] for row in rows]),
        "values": [row["triggered_steps"] for row in rows],
    }
    return aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fraction(values: list[float], predicate: Any) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if predicate(value)) / len(values)


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _nullable_sub(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_line(line: str) -> dict[str, Any]:
    return json.loads(line)
