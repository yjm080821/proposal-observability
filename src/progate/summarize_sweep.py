from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math
import statistics
import tomllib

from .io import write_json


def summarize_sweep(run_dirs: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(run_dir) for run_dir in run_dirs]
    rows = [row for row in rows if row is not None]

    csv_path = output_dir / "sweep_summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    write_json(
        output_dir / "sweep_summary.json",
        {
            "run_count": len(rows),
            "runs": rows,
            "aggregate": _aggregate(rows),
            "direction_note": "For future_val_delta = L_val(t+w) - L_val(t), lower is better. More negative correlation is better.",
        },
    )
    aggregate_rows = _aggregate_rows(rows)
    if aggregate_rows:
        with (output_dir / "sweep_aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(aggregate_rows))
            writer.writeheader()
            writer.writerows(aggregate_rows)
    return output_dir


def _summary_row(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.toml"
    analysis_path = run_dir / "analysis" / "future_windows_summary.json"
    if not summary_path.exists() or not config_path.exists() or not analysis_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = _read_toml(config_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    pilot0 = config.get("pilot0_lora", {})
    run = config.get("run", {})

    row: dict[str, Any] = {
        "run_id": run_dir.name,
        "name": run.get("name"),
        "seed": run.get("seed"),
        "probe_k": pilot0.get("probe_k"),
        "lambda_v": pilot0.get("lambda_v"),
        "steps": pilot0.get("steps"),
        "rows": summary.get("rows"),
        "loss_spikes": summary.get("loss_spikes"),
        "mean_probe_score": summary.get("mean_probe_score"),
        "mean_validation_loss": summary.get("mean_validation_loss"),
    }

    for key, value in sorted(analysis.items()):
        if not key.startswith("w"):
            continue
        window = key[1:]
        row[f"corr_w{window}"] = value.get("probe_score_future_val_delta_pearson")
        percentile = _read_json(run_dir / "analysis" / key / "accepted_rejected_summary.json").get("percentile", {})
        top = percentile.get("top", {})
        bottom = percentile.get("bottom", {})
        top_delta = top.get("future_val_delta_mean")
        bottom_delta = bottom.get("future_val_delta_mean")
        row[f"top20_delta_w{window}"] = top_delta
        row[f"bottom20_delta_w{window}"] = bottom_delta
        row[f"top_minus_bottom_w{window}"] = _safe_subtract(top_delta, bottom_delta)
    return row


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"K={row.get('probe_k')},lambda={row.get('lambda_v')}"
        grouped.setdefault(key, []).append(row)
    return {key: _aggregate_group(group) for key, group in grouped.items()}


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped = _aggregate(rows)
    for key, values in grouped.items():
        row = {"setting": key}
        row.update(values)
        output.append(row)
    return output


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        key
        for key in _fieldnames(rows)
        if key.startswith("corr_w") or key.startswith("top_minus_bottom_w")
    ]
    output: dict[str, Any] = {"run_count": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        output[f"{metric}_mean"] = _mean(values)
        output[f"{metric}_std"] = _std(values)
    return output


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_subtract(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    result = statistics.stdev(values)
    if math.isnan(result):
        return None
    return result
