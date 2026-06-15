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


def analyze_proposal_predictability(
    run_dir: Path,
    output_dir: Path | None = None,
    future_windows: list[int] | None = None,
    top_fractions: list[float] | None = None,
    permutation_n: int = 5000,
    seed: int = 20260511,
) -> Path:
    windows = future_windows or [1, 5, 10, 20]
    fractions = top_fractions or [0.1, 0.2, 0.3]
    output = output_dir or (run_dir / "analysis" / "proposal_predictability")
    output.mkdir(parents=True, exist_ok=True)

    run_dirs = _resolve_run_dirs(run_dir)
    rng = random.Random(seed)

    per_seed_by_window: dict[int, list[dict[str, Any]]] = {window: [] for window in windows}
    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []

    for candidate in run_dirs:
        raw_rows = read_jsonl(candidate / "metrics.jsonl")
        run_seed = _run_seed(raw_rows, candidate)
        for window in windows:
            window_rows = _valid_rows(rows_for_future_window(raw_rows, window))
            if not window_rows:
                continue
            per_seed_by_window[window].append(
                {
                    "seed": run_seed,
                    "run_id": candidate.name,
                    "rows": window_rows,
                }
            )
            metric_rows.append(_metric_row("per_seed", run_seed, candidate.name, window, window_rows))
            for fraction in fractions:
                gap_rows.append(
                    _gap_row(
                        "per_seed",
                        run_seed,
                        candidate.name,
                        window,
                        fraction,
                        [window_rows],
                        permutation_n,
                        rng,
                    )
                )

    for window, seed_groups in sorted(per_seed_by_window.items()):
        pooled_rows = [row for group in seed_groups for row in group["rows"]]
        if not pooled_rows:
            continue
        metric_rows.append(_metric_row("pooled", None, None, window, pooled_rows, seed_count=len(seed_groups)))
        for fraction in fractions:
            gap_rows.append(
                _gap_row(
                    "pooled_stratified",
                    None,
                    None,
                    window,
                    fraction,
                    [group["rows"] for group in seed_groups],
                    permutation_n,
                    rng,
                    seed_count=len(seed_groups),
                )
            )

    _write_csv(output / "predictability_metrics.csv", metric_rows)
    _write_csv(output / "top_fraction_gaps.csv", gap_rows)
    write_json(
        output / "proposal_predictability_summary.json",
        {
            "input_dir": str(run_dir),
            "run_ids": [path.name for path in run_dirs],
            "future_windows": windows,
            "top_fractions": fractions,
            "permutation_n": permutation_n,
            "permutation_seed": seed,
            "direction_note": (
                "future_val_delta = L_val(t+w) - L_val(t). Lower is better. "
                "Negative correlations and negative top-minus-bottom gaps support predictability. "
                "AUC is computed with future-good labels where future_val_delta is at or below the median; "
                "AUC above 0.5 supports predictability."
            ),
            "metrics": metric_rows,
            "top_fraction_gaps": gap_rows,
        },
    )
    return output


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
    if not output:
        raise ValueError(f"no runs found in {summary_path}")
    return output


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("probe_score") is not None and row.get("future_val_delta") is not None
    ]


def _run_seed(rows: list[dict[str, Any]], run_dir: Path) -> int | str | None:
    for row in rows:
        if row.get("seed") is not None:
            return row["seed"]
    marker = "_seed"
    if marker not in run_dir.name:
        return None
    suffix = run_dir.name.rsplit(marker, maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else suffix


def _metric_row(
    scope: str,
    seed: int | str | None,
    run_id: str | None,
    window: int,
    rows: list[dict[str, Any]],
    seed_count: int | None = None,
) -> dict[str, Any]:
    scores = [float(row["probe_score"]) for row in rows]
    deltas = [float(row["future_val_delta"]) for row in rows]
    labels = _future_good_labels(deltas)
    row = {
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
        shuffled_groups = [_shuffle_scores_within_group(group, rng) for group in row_groups]
        permuted.append(_stratified_gap(shuffled_groups, fraction))
    p_value = (sum(1 for value in permuted if value <= observed) + 1) / (permutation_n + 1)
    row = {
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


def _stratified_gap(row_groups: list[list[dict[str, Any]]], fraction: float) -> float:
    top: list[float] = []
    bottom: list[float] = []
    for rows in row_groups:
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: float(row["probe_score"]))
        count = max(1, int(round(len(ordered) * fraction)))
        bottom.extend(float(row["future_val_delta"]) for row in ordered[:count])
        top.extend(float(row["future_val_delta"]) for row in ordered[-count:])
    return statistics.fmean(top) - statistics.fmean(bottom)


def _shuffle_scores_within_group(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    scores = [row["probe_score"] for row in rows]
    rng.shuffle(scores)
    shuffled: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        new_row = dict(row)
        new_row["probe_score"] = score
        shuffled.append(new_row)
    return shuffled


def _future_good_labels(deltas: list[float]) -> list[int]:
    if not deltas:
        return []
    median = statistics.median(deltas)
    return [1 if delta <= median else 0 for delta in deltas]


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


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            dx = _sign(xs[i] - xs[j])
            dy = _sign(ys[i] - ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0.0:
        return None
    return (concordant - discordant) / denom


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _auc(scores: list[float], labels: list[int]) -> float | None:
    if len(scores) != len(labels) or not scores:
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


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
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names
