from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import math
import shutil
import statistics
import sys

from .config import AnalysisConfig, ExperimentConfig
from .io import write_json, write_jsonl
from .repro import environment_snapshot, set_seed
from .run_dir import create_run_dir


def run_pilot0(config: ExperimentConfig, config_path: Path, project_root: Path) -> Path:
    rng = set_seed(config.run.seed)
    run_dir = create_run_dir(config.run.output_dir, config.run.name, config.run.seed)

    shutil.copy2(config_path, run_dir / "config.toml")
    write_json(run_dir / "env.json", environment_snapshot(project_root, sys.argv))

    run_id = run_dir.name
    rows = _simulate(config, rng, run_id=run_id)
    summary = summarize_rows(rows, config.analysis)

    write_jsonl(run_dir / "metrics.jsonl", rows)
    write_json(run_dir / "summary.json", summary)
    _write_run_note(run_dir, config, summary)
    return run_dir


def summarize_rows(rows: list[dict[str, Any]], analysis: AnalysisConfig) -> dict[str, Any]:
    scored = [row for row in rows if row["future_val_delta"] is not None]
    accepted = [row for row in scored if row["alpha_probe"] >= analysis.alpha_acc]
    rejected = [row for row in scored if row["alpha_probe"] <= analysis.alpha_rej]

    bins = _score_bins(scored, analysis.bins)
    return {
        "rows": len(rows),
        "scored_rows": len(scored),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "mean_future_delta_accepted": _mean([row["future_val_delta"] for row in accepted]),
        "mean_future_delta_rejected": _mean([row["future_val_delta"] for row in rejected]),
        "probe_score_future_delta_correlation": _pearson(
            [row["probe_score"] for row in scored],
            [row["future_val_delta"] for row in scored],
        ),
        "loss_spikes": sum(1 for row in rows if row["loss_spike"]),
        "score_bins": bins,
    }


def _simulate(config: ExperimentConfig, rng: Any, run_id: str) -> list[dict[str, Any]]:
    cfg = config.pilot0
    phi = [rng.gauss(0.0, 0.2) for _ in range(cfg.dimension)]
    target = [rng.gauss(0.0, 1.0) for _ in range(cfg.dimension)]
    proposal_norm_ema = 0.0
    train_losses: list[float] = []
    val_losses: list[float] = []
    rows: list[dict[str, Any]] = []

    for step in range(cfg.steps):
        train_target = _sample_target(target, cfg.train_noise, rng)
        grad = [value - target_value for value, target_value in zip(phi, train_target, strict=True)]
        proposal = [-cfg.learning_rate * value for value in grad]
        proposal_norm = _norm(proposal)
        proposal_norm_ema = proposal_norm if step == 0 else 0.95 * proposal_norm_ema + 0.05 * proposal_norm

        train_loss = _loss(phi, train_target)
        val_target = _sample_target(target, cfg.validation_noise, rng)
        val_loss = _loss(phi, val_target)
        probe_delta = _probe_delta(phi, proposal, target, cfg.probe_noise, cfg.lambda_v, cfg.probe_k, rng)
        alpha_probe = _clip((-probe_delta - cfg.tau_delta) / cfg.gamma, 0.0, 1.0)
        loss_spike = bool(train_losses and train_loss - train_losses[-1] > cfg.loss_spike_threshold)

        rows.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "mode": "synthetic",
                "model_name": "synthetic-quadratic",
                "dataset_name": "synthetic-noisy-quadratic",
                "split_name": "train",
                "global_step": step,
                "optimizer_step": step,
                "epoch": 0,
                "lr": cfg.learning_rate,
                "lora_rank": None,
                "batch_id": f"train-{step}",
                "probe_batch_id": f"probe-{step}",
                "val_window_id": f"val-w{cfg.future_window}-{step}",
                "step": step,
                "seed": config.run.seed,
                "train_loss": train_loss,
                "validation_loss_snapshot": val_loss,
                "q_t_norm": proposal_norm,
                "q_t_norm_ema": proposal_norm_ema,
                "delta_bar": probe_delta,
                "probe_delta_bar": probe_delta,
                "probe_score": -probe_delta,
                "alpha_probe": alpha_probe,
                "lambda_v": cfg.lambda_v,
                "K": cfg.probe_k,
                "future_window_w": cfg.future_window,
                "future_window": cfg.future_window,
                "future_val_delta": None,
                "loss_spike": loss_spike,
            }
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        phi = [value + delta for value, delta in zip(phi, proposal, strict=True)]

    for index, row in enumerate(rows):
        future_index = index + cfg.future_window
        if future_index < len(val_losses):
            row["future_val_delta"] = val_losses[future_index] - val_losses[index]

    return rows


def _probe_delta(
    phi: list[float],
    proposal: list[float],
    target: list[float],
    noise: float,
    lambda_v: float,
    probe_k: int,
    rng: Any,
) -> float:
    shifted = [value + lambda_v * delta for value, delta in zip(phi, proposal, strict=True)]
    deltas = []
    for _ in range(probe_k):
        probe_target = _sample_target(target, noise, rng)
        deltas.append(_loss(shifted, probe_target) - _loss(phi, probe_target))
    return statistics.fmean(deltas)


def _sample_target(target: list[float], noise: float, rng: Any) -> list[float]:
    return [value + rng.gauss(0.0, noise) for value in target]


def _loss(phi: list[float], target: list[float]) -> float:
    return 0.5 * statistics.fmean((value - target_value) ** 2 for value, target_value in zip(phi, target, strict=True))


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


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


def _score_bins(rows: list[dict[str, Any]], bin_count: int) -> list[dict[str, Any]]:
    if not rows or bin_count <= 0:
        return []
    ordered = sorted(rows, key=lambda row: row["probe_score"])
    bins: list[dict[str, Any]] = []
    for index in range(bin_count):
        start = index * len(ordered) // bin_count
        end = (index + 1) * len(ordered) // bin_count
        chunk = ordered[start:end]
        bins.append(
            {
                "bin": index,
                "count": len(chunk),
                "score_min": chunk[0]["probe_score"] if chunk else None,
                "score_max": chunk[-1]["probe_score"] if chunk else None,
                "mean_probe_score": _mean([row["probe_score"] for row in chunk]),
                "mean_future_val_delta": _mean([row["future_val_delta"] for row in chunk]),
            }
        )
    return bins


def _write_run_note(run_dir: Path, config: ExperimentConfig, summary: dict[str, Any]) -> None:
    note = run_dir / "run_note.md"
    note.write_text(
        "\n".join(
            [
                f"# {config.run.name}",
                "",
                "## Purpose",
                "",
                "Validate the Pilot 0 logging path before attaching a real LLM fine-tuning loop.",
                "",
                "## Key Check",
                "",
                "Probe score should be predictive of held-out future validation delta.",
                "",
                "## Summary",
                "",
                f"- rows: {summary['rows']}",
                f"- scored rows: {summary['scored_rows']}",
                f"- accepted rows: {summary['accepted_rows']}",
                f"- rejected rows: {summary['rejected_rows']}",
                f"- score/future-delta correlation: {summary['probe_score_future_delta_correlation']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
