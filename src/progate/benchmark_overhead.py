from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math
import shutil
import statistics
import sys
import time
import tomllib

from .io import write_json
from .pilot0_lora import (
    Batch,
    LoraExperimentConfig,
    _loss,
    _make_batches,
    _resolve_device,
    _resolve_dtype,
    _split_rows,
    compute_adamw_proposal,
    compute_probe_delta,
    load_lora_config,
)
from .repro import environment_snapshot, set_seed
from .run_dir import create_run_dir


def run_overhead_benchmark(config: LoraExperimentConfig, config_path: Path, project_root: Path) -> Path:
    raw = _read_toml(config_path)
    benchmark = raw.get("benchmark", {})
    warmup_steps = int(benchmark.get("warmup_steps", 10))
    measure_steps = int(benchmark.get("measure_steps", 50))
    probe_ks = [int(value) for value in benchmark.get("probe_ks", [0, 1, 2, 4])]

    set_seed(config.run.seed)
    run_dir = create_run_dir(config.run.output_dir, f"{config.run.name}-overhead", config.run.seed)
    shutil.copy2(config_path, run_dir / "config.toml")
    write_json(run_dir / "env.json", environment_snapshot(project_root, sys.argv))

    result = _run_benchmark(config, probe_ks, warmup_steps, measure_steps, run_dir)
    _write_csv(run_dir / "benchmark_results.csv", result["summary_rows"])
    _write_jsonl(run_dir / "benchmark_steps.jsonl", result["step_rows"])
    write_json(run_dir / "summary.json", result["summary"])
    _write_run_note(run_dir, config, result["summary"])
    return run_dir


def _run_benchmark(
    config: LoraExperimentConfig,
    probe_ks: list[int],
    warmup_steps: int,
    measure_steps: int,
    run_dir: Path,
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(torch, config.model.device)
    dtype = _resolve_dtype(torch, config.model.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(config.dataset.name, split=config.dataset.split)
    dataset = dataset.shuffle(seed=config.run.seed)
    needed = config.dataset.train_size + config.dataset.probe_size + config.dataset.validation_size
    dataset = dataset.select(range(min(needed, len(dataset))))
    train_rows, probe_rows, _ = _split_rows(dataset, config)
    train_batches = _make_batches(train_rows, tokenizer, config, config.dataset.train_batch_size, "train", device)
    probe_batches = _make_batches(probe_rows, tokenizer, config, config.dataset.probe_batch_size, "probe", device)
    if not train_batches or not probe_batches:
        raise RuntimeError("train and probe streams must be non-empty")

    step_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for probe_k in probe_ks:
        setting_rows = _benchmark_setting(
            config=config,
            torch=torch,
            peft_config_class=PeftLoraConfig,
            get_peft_model=get_peft_model,
            model_class=AutoModelForCausalLM,
            load_dtype=dtype,
            device=device,
            probe_k=probe_k,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            train_batches=train_batches,
            probe_batches=probe_batches,
        )
        step_rows.extend(setting_rows)
        summary_rows.append(_summarize_setting(setting_rows, probe_k, device, torch))

    baseline = next((row for row in summary_rows if int(row["probe_k"]) == 0), None)
    if baseline is not None and baseline.get("mean_step_time_s"):
        base_time = float(baseline["mean_step_time_s"])
        for row in summary_rows:
            row["relative_step_time"] = float(row["mean_step_time_s"]) / base_time
            row["relative_overhead_pct"] = 100.0 * (float(row["mean_step_time_s"]) / base_time - 1.0)

    return {
        "step_rows": step_rows,
        "summary_rows": summary_rows,
        "summary": {
            "mode": "overhead_benchmark",
            "model_name": config.model.name,
            "dataset_name": config.dataset.name,
            "seed": config.run.seed,
            "device": device,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "probe_ks": probe_ks,
            "direction_note": "probe_k=0 is plain AdamW. probe_k>0 uses current ProGate diagnostic implementation: AdamW proposal extraction plus current/virtual probe forwards.",
            "results": summary_rows,
        },
    }


def _benchmark_setting(
    config: LoraExperimentConfig,
    torch: Any,
    peft_config_class: Any,
    get_peft_model: Any,
    model_class: Any,
    load_dtype: Any,
    device: str,
    probe_k: int,
    warmup_steps: int,
    measure_steps: int,
    train_batches: list[Batch],
    probe_batches: list[Batch],
) -> list[dict[str, Any]]:
    set_seed(config.run.seed + probe_k)
    load_kwargs: dict[str, Any] = {"trust_remote_code": config.model.trust_remote_code}
    if load_dtype is not None:
        load_kwargs["dtype"] = load_dtype
    model = model_class.from_pretrained(config.model.name, **load_kwargs)
    model.to(device)
    model.train()
    peft_config = peft_config_class(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        target_modules=config.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.train()
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, Any]] = []
    total_steps = warmup_steps + measure_steps
    for local_step in range(total_steps):
        measured = local_step >= warmup_steps
        train_batch = train_batches[local_step % len(train_batches)]
        selected_probe_batches = [
            probe_batches[(local_step * max(probe_k, 1) + index) % len(probe_batches)]
            for index in range(probe_k)
        ]
        _sync(torch, device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        train_loss = _loss(model, train_batch.tensors)
        train_loss.backward()
        q_t_norm = None
        delta_bar = None
        if probe_k > 0:
            proposal = compute_adamw_proposal(optimizer, trainable, torch)
            q_t_norm = proposal.q_t_norm
            delta_bar = compute_probe_delta(
                model,
                trainable,
                proposal,
                selected_probe_batches,
                config.pilot0_lora.lambda_v,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync(torch, device)
        elapsed = time.perf_counter() - start
        if measured:
            rows.append(
                {
                    "probe_k": probe_k,
                    "step_index": local_step - warmup_steps,
                    "elapsed_s": elapsed,
                    "train_loss": float(train_loss.detach().cpu().item()),
                    "q_t_norm": q_t_norm,
                    "delta_bar": delta_bar,
                    "probe_forward_evaluations": 2 * probe_k,
                    "peak_memory_allocated_bytes": _peak_memory(torch, device, "allocated"),
                    "peak_memory_reserved_bytes": _peak_memory(torch, device, "reserved"),
                }
            )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def _summarize_setting(rows: list[dict[str, Any]], probe_k: int, device: str, torch: Any) -> dict[str, Any]:
    elapsed = [float(row["elapsed_s"]) for row in rows]
    return {
        "probe_k": probe_k,
        "probe_forward_evaluations": 2 * probe_k,
        "steps": len(rows),
        "mean_step_time_s": statistics.fmean(elapsed),
        "std_step_time_s": statistics.stdev(elapsed) if len(elapsed) > 1 else None,
        "median_step_time_s": statistics.median(elapsed),
        "min_step_time_s": min(elapsed),
        "max_step_time_s": max(elapsed),
        "examples_per_second": 1.0 / statistics.fmean(elapsed) if elapsed else None,
        "peak_memory_allocated_gb": _bytes_to_gb(_peak_memory(torch, device, "allocated")),
        "peak_memory_reserved_gb": _bytes_to_gb(_peak_memory(torch, device, "reserved")),
    }


def _sync(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _peak_memory(torch: Any, device: str, kind: str) -> int | None:
    if device != "cuda":
        return None
    if kind == "allocated":
        return int(torch.cuda.max_memory_allocated())
    if kind == "reserved":
        return int(torch.cuda.max_memory_reserved())
    raise ValueError(f"unknown memory kind: {kind}")


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024**3)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def _write_run_note(run_dir: Path, config: LoraExperimentConfig, summary: dict[str, Any]) -> None:
    lines = [
        f"# {config.run.name} overhead benchmark",
        "",
        "## Purpose",
        "",
        "Measure step-time overhead for the current forward-only proposal diagnostic implementation.",
        "",
        "## Important Interpretation",
        "",
        "Probe K counts probe batches. The current implementation evaluates each probe batch before and after the virtual proposal, so it performs 2K probe forward evaluations per measured proposal.",
        "",
        "## Summary",
        "",
    ]
    for row in summary["results"]:
        lines.append(
            f"- K={row['probe_k']}: mean {row['mean_step_time_s']:.4f}s, "
            f"relative {row.get('relative_step_time', float('nan')):.3f}x, "
            f"overhead {row.get('relative_overhead_pct', float('nan')):.1f}%"
        )
    lines.extend(["", "## Files", "", "- benchmark_results.csv", "- benchmark_steps.jsonl", "- summary.json", ""])
    (run_dir / "run_note.md").write_text("\n".join(lines), encoding="utf-8")
