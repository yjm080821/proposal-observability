from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import csv
import random
import statistics

from .analyze_pilot0 import rows_for_future_window
from .io import read_jsonl, write_json
from .robustness import (
    _auc,
    _future_good_labels,
    _kendall_tau_b,
    _pearson,
    _resolve_run_dirs,
    _run_seed,
    _spearman,
)


Transform = Callable[[float], float]


def analyze_diagnostic_baselines(
    run_dir: Path,
    output_dir: Path | None = None,
    future_windows: list[int] | None = None,
    top_fraction: float = 0.2,
    permutation_n: int = 5000,
    seed: int = 20260511,
) -> Path:
    windows = future_windows or [1, 5, 10, 20]
    output = output_dir or (run_dir / "analysis" / "diagnostic_baselines")
    output.mkdir(parents=True, exist_ok=True)

    run_dirs = _resolve_run_dirs(run_dir)
    rng = random.Random(seed)
    baseline_specs = _baseline_specs(seed)

    per_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for candidate in run_dirs:
        raw_rows = read_jsonl(candidate / "metrics.jsonl")
        run_seed = _run_seed(raw_rows, candidate)
        for window in windows:
            base_window_rows = rows_for_future_window(raw_rows, window)
            for spec in baseline_specs:
                scored_rows = _scored_rows(base_window_rows, spec)
                coverage_rows.append(
                    {
                        "score_name": spec["name"],
                        "score_field": spec["field"],
                        "scope": "per_seed",
                        "seed": run_seed,
                        "run_id": candidate.name,
                        "future_window": window,
                        "available_n": len(scored_rows),
                        "candidate_n": len(base_window_rows),
                    }
                )
                if not scored_rows:
                    continue
                key = (spec["name"], window)
                per_seed.setdefault(key, []).append(
                    {
                        "seed": run_seed,
                        "run_id": candidate.name,
                        "rows": scored_rows,
                    }
                )
                metric_rows.append(
                    _metric_row(spec, "per_seed", run_seed, candidate.name, window, scored_rows)
                )
                gap_rows.append(
                    _gap_row(
                        spec,
                        "per_seed",
                        run_seed,
                        candidate.name,
                        window,
                        top_fraction,
                        [scored_rows],
                        permutation_n,
                        rng,
                    )
                )

    for (score_name, window), groups in sorted(per_seed.items()):
        spec = next(item for item in baseline_specs if item["name"] == score_name)
        pooled = [row for group in groups for row in group["rows"]]
        if not pooled:
            continue
        metric_rows.append(
            _metric_row(spec, "pooled", None, None, window, pooled, seed_count=len(groups))
        )
        gap_rows.append(
            _gap_row(
                spec,
                "pooled_stratified",
                None,
                None,
                window,
                top_fraction,
                [group["rows"] for group in groups],
                permutation_n,
                rng,
                seed_count=len(groups),
            )
        )

    best_oriented_rows = _best_oriented_rows(metric_rows, gap_rows, windows)

    _write_csv(output / "baseline_predictability_metrics.csv", metric_rows)
    _write_csv(output / "baseline_top_bottom_gaps.csv", gap_rows)
    _write_csv(output / "baseline_best_oriented_metrics.csv", best_oriented_rows)
    _write_csv(output / "baseline_coverage.csv", coverage_rows)
    write_json(
        output / "diagnostic_baselines_summary.json",
        {
            "input_dir": str(run_dir),
            "run_ids": [path.name for path in run_dirs],
            "future_windows": windows,
            "top_fraction": top_fraction,
            "permutation_n": permutation_n,
            "permutation_seed": seed,
            "direction_note": (
                "future_val_delta = L_val(t+w) - L_val(t). Lower is better. "
                "All baseline scores are oriented so larger means the proxy predicts a better proposal. "
                "Negative correlations and negative top-minus-bottom gaps support predictability; "
                "AUC above 0.5 supports predictability."
            ),
            "metrics": metric_rows,
            "top_bottom_gaps": gap_rows,
            "best_oriented_metrics": best_oriented_rows,
            "coverage": coverage_rows,
        },
    )
    return output


def _baseline_specs(seed: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "ProGate score",
            "group": "ProGate score",
            "field": "probe_score",
            "transform": _identity,
            "orientation": "higher_probe_score_is_better",
            "selection": "fixed",
        },
        {
            "name": "Proposal norm (larger)",
            "group": "Proposal norm",
            "field": "q_t_norm",
            "transform": _identity,
            "orientation": "larger_norm_is_better",
            "selection": "candidate",
        },
        {
            "name": "Proposal norm (smaller)",
            "group": "Proposal norm",
            "field": "q_t_norm",
            "transform": _negate,
            "orientation": "smaller_norm_is_better",
            "selection": "candidate",
        },
        {
            "name": "Gradient norm (larger)",
            "group": "Gradient norm",
            "field": "grad_norm",
            "transform": _identity,
            "orientation": "larger_norm_is_better",
            "selection": "candidate",
        },
        {
            "name": "Gradient norm (smaller)",
            "group": "Gradient norm",
            "field": "grad_norm",
            "transform": _negate,
            "orientation": "smaller_norm_is_better",
            "selection": "candidate",
        },
        {
            "name": "Train loss (larger)",
            "group": "Train loss",
            "field": "train_loss",
            "transform": _identity,
            "orientation": "larger_loss_is_better",
            "selection": "candidate",
        },
        {
            "name": "Train loss (smaller)",
            "group": "Train loss",
            "field": "train_loss",
            "transform": _negate,
            "orientation": "smaller_loss_is_better",
            "selection": "candidate",
        },
        {
            "name": "Probe loss before (larger)",
            "group": "Probe loss before",
            "field": "probe_loss_before",
            "transform": _identity,
            "orientation": "larger_probe_loss_is_better",
            "selection": "candidate",
        },
        {
            "name": "Probe loss before (smaller)",
            "group": "Probe loss before",
            "field": "probe_loss_before",
            "transform": _negate,
            "orientation": "smaller_probe_loss_is_better",
            "selection": "candidate",
        },
        {
            "name": "Probe loss after (larger)",
            "group": "Probe loss after",
            "field": "probe_loss_after",
            "transform": _identity,
            "orientation": "larger_probe_loss_is_better",
            "selection": "candidate",
        },
        {
            "name": "Probe loss after (smaller)",
            "group": "Probe loss after",
            "field": "probe_loss_after",
            "transform": _negate,
            "orientation": "smaller_probe_loss_is_better",
            "selection": "candidate",
        },
        {
            "name": "Random",
            "group": "Random",
            "field": "__random__",
            "transform": _identity,
            "orientation": "random_uniform",
            "selection": "fixed",
            "seed": seed,
        },
    ]


def _scored_rows(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    field = str(spec["field"])
    for row in rows:
        if row.get("future_val_delta") is None:
            continue
        if field == "__random__":
            raw = _stable_random_score(row, int(spec["seed"]))
        else:
            if row.get(field) is None:
                continue
            raw = float(row[field])
        score = spec["transform"](float(raw))
        new_row = dict(row)
        new_row["diagnostic_score"] = score
        scored.append(new_row)
    return scored


def _metric_row(
    spec: dict[str, Any],
    scope: str,
    seed: int | str | None,
    run_id: str | None,
    window: int,
    rows: list[dict[str, Any]],
    seed_count: int | None = None,
) -> dict[str, Any]:
    scores = [float(row["diagnostic_score"]) for row in rows]
    deltas = [float(row["future_val_delta"]) for row in rows]
    labels = _future_good_labels(deltas)
    row = {
        "score_name": spec["name"],
        "score_group": spec["group"],
        "score_field": spec["field"],
        "orientation": spec["orientation"],
        "selection": spec["selection"],
        "scope": scope,
        "seed": seed,
        "run_id": run_id,
        "future_window": window,
        "n": len(rows),
        "pearson": _pearson(scores, deltas),
        "spearman": _spearman(scores, deltas),
        "kendall_tau_b": _kendall_tau_b(scores, deltas),
        "proposal_predictive_auc": _auc(scores, labels),
        "future_good_count": sum(labels),
        "future_bad_count": len(labels) - sum(labels),
    }
    if seed_count is not None:
        row["seed_count"] = seed_count
    return row


def _gap_row(
    spec: dict[str, Any],
    scope: str,
    seed: int | str | None,
    run_id: str | None,
    window: int,
    fraction: float,
    row_groups: list[list[dict[str, Any]]],
    permutation_n: int,
    rng: random.Random,
    seed_count: int | None = None,
) -> dict[str, Any]:
    observed = _stratified_gap(row_groups, fraction)
    permuted: list[float] = []
    for _ in range(permutation_n):
        shuffled = [_shuffle_scores(group, rng) for group in row_groups]
        permuted.append(_stratified_gap(shuffled, fraction))
    p_value = (sum(1 for value in permuted if value <= observed) + 1) / (permutation_n + 1)
    row = {
        "score_name": spec["name"],
        "score_group": spec["group"],
        "score_field": spec["field"],
        "orientation": spec["orientation"],
        "selection": spec["selection"],
        "scope": scope,
        "seed": seed,
        "run_id": run_id,
        "future_window": window,
        "top_fraction": fraction,
        "n": sum(len(group) for group in row_groups),
        "top_minus_bottom_gap": observed,
        "permutation_p_one_sided": p_value,
        "permutation_n": permutation_n,
    }
    if seed_count is not None:
        row["seed_count"] = seed_count
    return row


def _best_oriented_rows(
    metric_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    windows: list[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    gap_lookup = {
        (
            row["score_name"],
            row["future_window"],
            row["scope"],
            row.get("seed"),
            row.get("run_id"),
        ): row
        for row in gap_rows
    }

    pooled_metrics = [
        row for row in metric_rows if row.get("scope") == "pooled" and row.get("future_window") in windows
    ]
    groups = sorted({str(row["score_group"]) for row in pooled_metrics})

    for group in groups:
        for window in windows:
            candidates = [
                row
                for row in pooled_metrics
                if row["score_group"] == group and row["future_window"] == window
            ]
            if not candidates:
                continue
            if group in {"ProGate score", "Random"}:
                chosen = candidates[0]
                selection_rule = "fixed_orientation"
            else:
                # Give simple scalar proxies their most favorable sign by AUC.
                # This makes the comparison conservative for ProGate.
                chosen = max(
                    candidates,
                    key=lambda row: (
                        float(row["proposal_predictive_auc"])
                        if row.get("proposal_predictive_auc") is not None
                        else float("-inf"),
                        -abs(float(row.get("spearman") or 0.0)),
                    ),
                )
                selection_rule = "best_auc_orientation_per_window"
            gap = gap_lookup.get((chosen["score_name"], window, "pooled_stratified", None, None), {})
            output.append(
                {
                    "score_group": group,
                    "selected_score_name": chosen["score_name"],
                    "selected_orientation": chosen["orientation"],
                    "selection_rule": selection_rule,
                    "future_window": window,
                    "n": chosen["n"],
                    "seed_count": chosen.get("seed_count"),
                    "pearson": chosen.get("pearson"),
                    "spearman": chosen.get("spearman"),
                    "kendall_tau_b": chosen.get("kendall_tau_b"),
                    "proposal_predictive_auc": chosen.get("proposal_predictive_auc"),
                    "top_minus_bottom_gap": gap.get("top_minus_bottom_gap"),
                    "permutation_p_one_sided": gap.get("permutation_p_one_sided"),
                }
            )

    return output


def _stratified_gap(row_groups: list[list[dict[str, Any]]], fraction: float) -> float:
    top: list[float] = []
    bottom: list[float] = []
    for rows in row_groups:
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: float(row["diagnostic_score"]))
        count = max(1, int(round(len(ordered) * fraction)))
        bottom.extend(float(row["future_val_delta"]) for row in ordered[:count])
        top.extend(float(row["future_val_delta"]) for row in ordered[-count:])
    return statistics.fmean(top) - statistics.fmean(bottom)


def _shuffle_scores(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    scores = [row["diagnostic_score"] for row in rows]
    rng.shuffle(scores)
    shuffled: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        new_row = dict(row)
        new_row["diagnostic_score"] = score
        shuffled.append(new_row)
    return shuffled


def _stable_random_score(row: dict[str, Any], seed: int) -> float:
    key = f"{seed}:{row.get('run_id')}:{row.get('seed')}:{row.get('step')}"
    return random.Random(key).random()


def _identity(value: float) -> float:
    return value


def _negate(value: float) -> float:
    return -value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
