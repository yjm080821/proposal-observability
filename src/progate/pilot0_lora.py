from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json
import math
import random
import shutil
import statistics
import sys
import tomllib

from .io import write_json
from .repro import environment_snapshot, set_seed
from .run_dir import create_run_dir


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    trust_remote_code: bool
    device: str
    model_class: str
    dtype: str
    device_map: str
    low_cpu_mem_usage: bool
    local_files_only: bool


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str
    train_size: int
    probe_size: int
    validation_size: int
    max_length: int
    train_batch_size: int
    probe_batch_size: int
    validation_batch_size: int
    instruction_field: str
    input_field: str
    output_field: str


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    exclude_modules: str | None


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float


@dataclass(frozen=True)
class Pilot0LoraConfig:
    steps: int
    lambda_v: float
    probe_k: int
    tau_delta: float
    gamma: float
    validation_interval: int
    loss_spike_threshold: float
    future_windows: list[int]
    update_mode: str
    verification_horizon_fraction: float
    alpha_min: float
    accepted_like_threshold: float
    cga_mode: str
    cga_trigger_fraction: float
    cga_warmup_steps: int
    cga_score_window: int
    cga_record_always_ga2: bool
    cga_improvement_margin: float
    control_mode: str
    control_trigger_fraction: float
    control_warmup_steps: int
    control_score_window: int
    control_damp_factor: float
    control_fixed_period: int
    control_verify_improvement: bool
    control_improvement_margin: float
    control_extra_policy: str


@dataclass(frozen=True)
class LoraExperimentConfig:
    run: RunConfig
    model: ModelConfig
    dataset: DatasetConfig
    lora: LoraConfig
    optimizer: OptimizerConfig
    pilot0_lora: Pilot0LoraConfig


@dataclass
class Batch:
    batch_id: str
    tensors: dict[str, Any]


@dataclass
class Proposal:
    updates: list[Any]
    snapshot: list[Any]
    q_t_norm: float


def load_lora_config(path: Path) -> LoraExperimentConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    run = raw.get("run", {})
    model = raw.get("model", {})
    dataset = raw.get("dataset", {})
    lora = raw.get("lora", {})
    optimizer = raw.get("optimizer", {})
    pilot0 = raw.get("pilot0_lora", {})

    return LoraExperimentConfig(
        run=RunConfig(
            name=str(_required(run, "name")),
            seed=int(_required(run, "seed")),
            output_dir=Path(str(_required(run, "output_dir"))),
        ),
        model=ModelConfig(
            name=str(_required(model, "name")),
            trust_remote_code=bool(_required(model, "trust_remote_code")),
            device=str(_required(model, "device")),
            model_class=str(model.get("model_class", "causal-lm")),
            dtype=str(_required(model, "dtype")),
            device_map=str(model.get("device_map", "none")),
            low_cpu_mem_usage=bool(model.get("low_cpu_mem_usage", False)),
            local_files_only=bool(model.get("local_files_only", False)),
        ),
        dataset=DatasetConfig(
            name=str(_required(dataset, "name")),
            split=str(_required(dataset, "split")),
            train_size=int(_required(dataset, "train_size")),
            probe_size=int(_required(dataset, "probe_size")),
            validation_size=int(_required(dataset, "validation_size")),
            max_length=int(_required(dataset, "max_length")),
            train_batch_size=int(_required(dataset, "train_batch_size")),
            probe_batch_size=int(_required(dataset, "probe_batch_size")),
            validation_batch_size=int(_required(dataset, "validation_batch_size")),
            instruction_field=str(_required(dataset, "instruction_field")),
            input_field=str(_required(dataset, "input_field")),
            output_field=str(_required(dataset, "output_field")),
        ),
        lora=LoraConfig(
            rank=int(_required(lora, "rank")),
            alpha=int(_required(lora, "alpha")),
            dropout=float(_required(lora, "dropout")),
            target_modules=[str(value) for value in _required(lora, "target_modules")],
            exclude_modules=str(lora["exclude_modules"]) if lora.get("exclude_modules") is not None else None,
        ),
        optimizer=OptimizerConfig(
            lr=float(_required(optimizer, "lr")),
            weight_decay=float(_required(optimizer, "weight_decay")),
            betas=tuple(float(value) for value in _required(optimizer, "betas")),  # type: ignore[arg-type]
            eps=float(_required(optimizer, "eps")),
        ),
        pilot0_lora=Pilot0LoraConfig(
            steps=int(_required(pilot0, "steps")),
            lambda_v=float(_required(pilot0, "lambda_v")),
            probe_k=int(_required(pilot0, "probe_k")),
            tau_delta=float(_required(pilot0, "tau_delta")),
            gamma=float(_required(pilot0, "gamma")),
            validation_interval=int(_required(pilot0, "validation_interval")),
            loss_spike_threshold=float(_required(pilot0, "loss_spike_threshold")),
            future_windows=[int(value) for value in _required(pilot0, "future_windows")],
            update_mode=str(pilot0.get("update_mode", "adamw")),
            verification_horizon_fraction=float(pilot0.get("verification_horizon_fraction", 0.0)),
            alpha_min=float(pilot0.get("alpha_min", 0.0)),
            accepted_like_threshold=float(pilot0.get("accepted_like_threshold", 0.8)),
            cga_mode=str(pilot0.get("cga_mode", "off")),
            cga_trigger_fraction=float(pilot0.get("cga_trigger_fraction", 0.3)),
            cga_warmup_steps=int(pilot0.get("cga_warmup_steps", 20)),
            cga_score_window=int(pilot0.get("cga_score_window", 50)),
            cga_record_always_ga2=bool(pilot0.get("cga_record_always_ga2", False)),
            cga_improvement_margin=float(pilot0.get("cga_improvement_margin", 0.0)),
            control_mode=str(pilot0.get("control_mode", "off")),
            control_trigger_fraction=float(pilot0.get("control_trigger_fraction", 0.25)),
            control_warmup_steps=int(pilot0.get("control_warmup_steps", 30)),
            control_score_window=int(pilot0.get("control_score_window", 50)),
            control_damp_factor=float(pilot0.get("control_damp_factor", 0.5)),
            control_fixed_period=int(pilot0.get("control_fixed_period", 4)),
            control_verify_improvement=bool(pilot0.get("control_verify_improvement", False)),
            control_improvement_margin=float(pilot0.get("control_improvement_margin", 0.0)),
            control_extra_policy=str(pilot0.get("control_extra_policy", "average_grad")),
        ),
    )


def run_pilot0_lora(config: LoraExperimentConfig, config_path: Path, project_root: Path) -> Path:
    set_seed(config.run.seed)
    run_dir = create_run_dir(config.run.output_dir, config.run.name, config.run.seed)
    shutil.copy2(config_path, run_dir / "config.toml")
    write_json(run_dir / "env.json", environment_snapshot(project_root, sys.argv))

    try:
        result = _run_training(config, run_dir)
    except KeyboardInterrupt:
        _write_interrupted(run_dir, "KeyboardInterrupt")
        _write_interrupted_run_note(run_dir, config)
        raise
    except Exception as error:
        _write_interrupted(run_dir, f"{type(error).__name__}: {error}")
        _write_interrupted_run_note(run_dir, config)
        raise

    write_json(run_dir / "summary.json", result["summary"])
    _write_run_note(run_dir, config, result["summary"])
    return run_dir


def _run_training(config: LoraExperimentConfig, run_dir: Path) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    import transformers

    _validate_pilot0_lora_config(config.pilot0_lora)

    device = _resolve_device(torch, config.model.device)
    dtype = _resolve_dtype(torch, config.model.dtype, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(config.dataset.name, split=config.dataset.split)
    dataset = dataset.shuffle(seed=config.run.seed)
    needed = config.dataset.train_size + config.dataset.probe_size + config.dataset.validation_size
    dataset = dataset.select(range(min(needed, len(dataset))))

    train_rows, probe_rows, validation_rows = _split_rows(dataset, config)
    train_batches = _make_batches(train_rows, tokenizer, config, config.dataset.train_batch_size, "train", device)
    probe_batches = _make_batches(probe_rows, tokenizer, config, config.dataset.probe_batch_size, "probe", device)
    validation_batches = _make_batches(
        validation_rows,
        tokenizer,
        config,
        config.dataset.validation_batch_size,
        "validation",
        device,
    )
    if not train_batches or not probe_batches or not validation_batches:
        raise RuntimeError("train, probe, and validation streams must all be non-empty")

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": config.model.local_files_only,
    }
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    device_map = _effective_device_map(config.model)
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if config.model.low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    model, model_load_class = _load_model(transformers, config.model, load_kwargs)
    if not _uses_device_map(config.model):
        model.to(device)
    model.train()

    lora_exclude_modules = _lora_exclude_modules(config.lora, model_load_class)
    peft_config = PeftLoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        target_modules=config.lora.target_modules,
        exclude_modules=lora_exclude_modules,
        task_type=None if model_load_class == "AutoModelForImageTextToText" else "CAUSAL_LM",
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

    run_id = run_dir.name
    rows: list[dict[str, Any]] = []
    train_losses: list[float] = []
    probe_score_history: list[float] = []
    control_rng = random.Random(config.run.seed + 104729)
    cumulative_backward_count = 0
    cumulative_extra_backward_count = 0
    q_t_norm_ema = 0.0
    horizon_steps = _verification_horizon_steps(config.pilot0_lora)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    with metrics_path.open("a", encoding="utf-8") as metrics_handle:
        for step in range(config.pilot0_lora.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_batch = train_batches[step % len(train_batches)]
            train_loss_tensor = _loss(model, train_batch.tensors)
            train_loss_tensor.backward()
            train_loss = _to_float(train_loss_tensor)
            train_grad_norm = grad_norm(trainable, torch)
            first_grads = snapshot_grads(trainable)

            proposal = compute_adamw_proposal(optimizer, trainable, torch)
            q_t_norm_ema = proposal.q_t_norm if step == 0 else 0.95 * q_t_norm_ema + 0.05 * proposal.q_t_norm

            selected_probe_batches = [
                probe_batches[(step * config.pilot0_lora.probe_k + index) % len(probe_batches)]
                for index in range(config.pilot0_lora.probe_k)
            ]
            probe_stats = compute_probe_stats(
                model,
                trainable,
                proposal,
                selected_probe_batches,
                config.pilot0_lora.lambda_v,
            )
            delta_bar = probe_stats["delta"]
            alpha_probe = _clip((-delta_bar - config.pilot0_lora.tau_delta) / config.pilot0_lora.gamma, 0.0, 1.0)
            horizon_active = config.pilot0_lora.update_mode == "progate_lite" and step < horizon_steps
            alpha_t = _effective_alpha(alpha_probe, horizon_active, config.pilot0_lora.alpha_min)
            effective_q_norm = alpha_t * proposal.q_t_norm
            cga_result = _maybe_run_cga_diagnostic(
                model=model,
                optimizer=optimizer,
                params=trainable,
                torch=torch,
                config=config,
                step=step,
                score_history=probe_score_history,
                first_grads=first_grads,
                train_batches=train_batches,
                selected_probe_batches=selected_probe_batches,
                first_probe_score=-delta_bar,
                first_q_norm=proposal.q_t_norm,
            )
            control_result = _maybe_run_scsa_control(
                model=model,
                optimizer=optimizer,
                params=trainable,
                torch=torch,
                config=config,
                step=step,
                score_history=probe_score_history,
                first_grads=first_grads,
                train_batches=train_batches,
                selected_probe_batches=selected_probe_batches,
                first_probe_score=-delta_bar,
                first_q_norm=proposal.q_t_norm,
                rng=control_rng,
            )
            row_backward_extra = int(control_result["backward_count_extra"])
            if config.pilot0_lora.control_mode == "off":
                row_backward_extra = int(cga_result.get("extra_backward_count") or 0)
            row_backward_total = 1 + row_backward_extra
            cumulative_backward_count += row_backward_total
            cumulative_extra_backward_count += row_backward_extra
            control_result = dict(control_result)
            control_result.update(
                {
                    "backward_count_main": 1,
                    "backward_count_extra": row_backward_extra,
                    "backward_count_total": row_backward_total,
                    "cumulative_backward_count": cumulative_backward_count,
                    "cumulative_extra_backward_count": cumulative_extra_backward_count,
                    "extra_backward_budget_fraction": cumulative_extra_backward_count / (step + 1),
                }
            )
            control_effective_q_norm = control_result.get("effective_update_norm")
            if control_effective_q_norm is None:
                control_effective_q_norm = effective_q_norm

            validation_loss = None
            if step % config.pilot0_lora.validation_interval == 0:
                validation_loss = evaluate_loss(model, validation_batches)

            loss_spike = bool(train_losses and train_loss - train_losses[-1] > config.pilot0_lora.loss_spike_threshold)
            row = {
                "schema_version": 1,
                "run_id": run_id,
                "mode": "real_lora",
                "update_mode": config.pilot0_lora.update_mode,
                "model_name": config.model.name,
                "model_load_class": model_load_class,
                "dataset_name": config.dataset.name,
                "split_name": config.dataset.split,
                "global_step": step,
                "optimizer_step": step,
                "epoch": 0,
                "lr": config.optimizer.lr,
                "lora_rank": config.lora.rank,
                "lora_exclude_modules": lora_exclude_modules,
                "batch_id": train_batch.batch_id,
                "probe_batch_id": ",".join(batch.batch_id for batch in selected_probe_batches),
                "val_window_id": "validation-stream",
                "step": step,
                "seed": config.run.seed,
                "train_loss": train_loss,
                "grad_norm": train_grad_norm,
                "validation_loss_snapshot": validation_loss,
                "q_t_norm": proposal.q_t_norm,
                "q_t_norm_ema": q_t_norm_ema,
                "effective_q_norm": control_effective_q_norm,
                "delta_bar": delta_bar,
                "probe_delta_bar": delta_bar,
                "probe_loss_before": probe_stats["base_loss"],
                "probe_loss_after": probe_stats["shifted_loss"],
                "probe_score": -delta_bar,
                "alpha_probe": alpha_probe,
                "alpha_t": alpha_t,
                "alpha_min": config.pilot0_lora.alpha_min,
                "accepted_like": alpha_t >= config.pilot0_lora.accepted_like_threshold,
                "dampened_like": alpha_t < config.pilot0_lora.accepted_like_threshold,
                "T_e_active": horizon_active,
                "verification_horizon_steps": horizon_steps,
                "lambda_v": config.pilot0_lora.lambda_v,
                "K": config.pilot0_lora.probe_k,
                "future_window_w": None,
                "future_window": None,
                "future_val_delta": None,
                "loss_spike": loss_spike,
                "backward_count_main": 1,
                "backward_count_extra": row_backward_extra,
                "backward_count_total": row_backward_total,
                "cumulative_backward_count": cumulative_backward_count,
                "cumulative_extra_backward_count": cumulative_extra_backward_count,
                "extra_backward_budget_fraction": cumulative_extra_backward_count / (step + 1),
            }
            row.update(cga_result)
            row.update(control_result)
            rows.append(row)
            json.dump(row, metrics_handle, ensure_ascii=False, sort_keys=True)
            metrics_handle.write("\n")
            metrics_handle.flush()

            optimizer.step()
            if config.pilot0_lora.update_mode == "progate_lite":
                set_params_to_scaled_proposal(trainable, proposal, alpha_t)
            elif config.pilot0_lora.control_mode == "scsa_damp" and control_result["control_triggered"]:
                set_params_to_scaled_proposal(trainable, proposal, config.pilot0_lora.control_damp_factor)
            elif config.pilot0_lora.update_mode not in {"adamw", "always_ga2", "progate_cga", "progate_cga_verified"}:
                raise ValueError(f"unknown update_mode: {config.pilot0_lora.update_mode}")
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(train_loss)
            probe_score_history.append(-delta_bar)

    summary = _summary(config, rows, trainable, device)
    return {"rows": rows, "summary": summary}


def compute_adamw_proposal(optimizer: Any, params: list[Any], torch: Any) -> Proposal:
    snapshot = snapshot_params(params)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    updates = [(param.detach() - before).detach().clone() for before, param in zip(snapshot, params, strict=True)]
    q_t_norm = vector_norm(updates, torch)
    restore_params(snapshot, params)
    optimizer.load_state_dict(optimizer_state)
    return Proposal(updates=updates, snapshot=snapshot, q_t_norm=q_t_norm)


def compute_probe_delta(
    model: Any,
    params: list[Any],
    proposal: Proposal,
    batches: list[Batch],
    lambda_v: float,
) -> float:
    return compute_probe_stats(model, params, proposal, batches, lambda_v)["delta"]


def compute_probe_stats(
    model: Any,
    params: list[Any],
    proposal: Proposal,
    batches: list[Batch],
    lambda_v: float,
) -> dict[str, float]:
    base_loss = evaluate_loss(model, batches)
    apply_vector_to_params(params, proposal.updates, lambda_v)
    shifted_loss = evaluate_loss(model, batches)
    restore_params(proposal.snapshot, params)
    return {
        "base_loss": base_loss,
        "shifted_loss": shifted_loss,
        "delta": shifted_loss - base_loss,
    }


def _maybe_run_scsa_control(
    model: Any,
    optimizer: Any,
    params: list[Any],
    torch: Any,
    config: LoraExperimentConfig,
    step: int,
    score_history: list[float],
    first_grads: list[Any | None],
    train_batches: list[Batch],
    selected_probe_batches: list[Batch],
    first_probe_score: float,
    first_q_norm: float,
    rng: random.Random,
) -> dict[str, Any]:
    mode = config.pilot0_lora.control_mode
    result = {
        "control_mode": mode,
        "control_triggered": False,
        "control_trigger_reason": "none",
        "control_threshold": None,
        "control_score_window_n": 0,
        "control_trigger_fraction": config.pilot0_lora.control_trigger_fraction,
        "control_warmup_steps": config.pilot0_lora.control_warmup_steps,
        "control_score_window": config.pilot0_lora.control_score_window,
        "control_fixed_period": config.pilot0_lora.control_fixed_period,
        "control_damp_factor": config.pilot0_lora.control_damp_factor,
        "control_verify_improvement": config.pilot0_lora.control_verify_improvement,
        "control_improvement_margin": config.pilot0_lora.control_improvement_margin,
        "commit_source": "q1",
        "commit_alpha": 1.0,
        "s1": first_probe_score,
        "s2": None,
        "score_improvement": None,
        "extra_batch_id": None,
        "control_q2_norm": None,
        "control_g2_norm": None,
        "effective_update_norm": first_q_norm,
        "backward_count_main": 1,
        "backward_count_extra": 0,
        "backward_count_total": 1,
    }
    if mode == "off":
        return result

    triggered, reason, threshold, window_n = _should_trigger_control(
        mode=mode,
        step=step,
        score=first_probe_score,
        score_history=score_history,
        fraction=config.pilot0_lora.control_trigger_fraction,
        warmup_steps=config.pilot0_lora.control_warmup_steps,
        window=config.pilot0_lora.control_score_window,
        fixed_period=config.pilot0_lora.control_fixed_period,
        rng=rng,
    )
    result.update(
        {
            "control_triggered": triggered,
            "control_trigger_reason": reason,
            "control_threshold": threshold,
            "control_score_window_n": window_n,
        }
    )
    if not triggered:
        return result

    if mode == "scsa_damp":
        result.update(
            {
                "commit_source": "q1_damped",
                "commit_alpha": config.pilot0_lora.control_damp_factor,
                "effective_update_norm": config.pilot0_lora.control_damp_factor * first_q_norm,
            }
        )
        return result

    if mode not in {"scsa_extra", "random_extra", "fixed_extra"}:
        raise ValueError(f"unknown control_mode: {mode}")

    extra_batch = train_batches[(step + 1) % len(train_batches)]
    extra = _build_extra_evidence_proposal(
        model=model,
        optimizer=optimizer,
        params=params,
        torch=torch,
        config=config,
        first_grads=first_grads,
        extra_batch=extra_batch,
        selected_probe_batches=selected_probe_batches,
        first_probe_score=first_probe_score,
        leave_average_grads=True,
    )
    verified_commit = True
    if config.pilot0_lora.control_verify_improvement:
        improvement = extra["score_improvement"]
        verified_commit = bool(improvement is not None and improvement > config.pilot0_lora.control_improvement_margin)
        if not verified_commit:
            restore_grads(first_grads, params)

    result.update(
        {
            "commit_source": "q2_extra" if verified_commit else "q1_verified_no_improve",
            "commit_alpha": 1.0,
            "s2": extra["score"],
            "score_improvement": extra["score_improvement"],
            "extra_batch_id": extra_batch.batch_id,
            "control_q2_norm": extra["q_norm"],
            "control_g2_norm": extra["g2_norm"],
            "effective_update_norm": extra["q_norm"] if verified_commit else first_q_norm,
            "backward_count_extra": 1,
            "backward_count_total": 2,
        }
    )
    return result


def _should_trigger_control(
    mode: str,
    step: int,
    score: float,
    score_history: list[float],
    fraction: float,
    warmup_steps: int,
    window: int,
    fixed_period: int,
    rng: random.Random,
) -> tuple[bool, str, float | None, int]:
    if step < warmup_steps:
        return False, "warmup", None, 0
    if mode in {"scsa_damp", "scsa_extra"}:
        recent_scores = _recent_scores(score_history, window)
        if not recent_scores:
            return False, "empty_score_window", None, 0
        threshold = _quantile(recent_scores, fraction)
        return score <= threshold, "score_quantile", threshold, len(recent_scores)
    if mode == "random_extra":
        return rng.random() < fraction, "random", None, 0
    if mode == "fixed_extra":
        return step % fixed_period == 0, "fixed", None, 0
    if mode == "off":
        return False, "none", None, 0
    raise ValueError(f"unknown control_mode: {mode}")


def _build_extra_evidence_proposal(
    model: Any,
    optimizer: Any,
    params: list[Any],
    torch: Any,
    config: LoraExperimentConfig,
    first_grads: list[Any | None],
    extra_batch: Batch,
    selected_probe_batches: list[Batch],
    first_probe_score: float,
    leave_average_grads: bool,
) -> dict[str, Any]:
    if config.pilot0_lora.control_extra_policy != "average_grad":
        raise ValueError(f"unknown control_extra_policy: {config.pilot0_lora.control_extra_policy}")

    optimizer.zero_grad(set_to_none=True)
    extra_loss = _loss(model, extra_batch.tensors)
    extra_loss.backward()
    g2_norm = grad_norm(params, torch)
    average_grads_with_snapshot(params, first_grads)
    proposal = compute_adamw_proposal(optimizer, params, torch)
    delta = compute_probe_delta(
        model,
        params,
        proposal,
        selected_probe_batches,
        config.pilot0_lora.lambda_v,
    )
    if not leave_average_grads:
        restore_grads(first_grads, params)
    score = -delta
    return {
        "score": score,
        "delta_bar": delta,
        "score_improvement": score - first_probe_score,
        "q_norm": proposal.q_t_norm,
        "g2_norm": g2_norm,
    }


def _maybe_run_cga_diagnostic(
    model: Any,
    optimizer: Any,
    params: list[Any],
    torch: Any,
    config: LoraExperimentConfig,
    step: int,
    score_history: list[float],
    first_grads: list[Any | None],
    train_batches: list[Batch],
    selected_probe_batches: list[Batch],
    first_probe_score: float,
    first_q_norm: float,
) -> dict[str, Any]:
    cga_mode = config.pilot0_lora.cga_mode
    result = {
        "cga_mode": cga_mode,
        "cga_commit_source": "q1",
        "cga_triggered": False,
        "cga_trigger_threshold": None,
        "cga_trigger_fraction": config.pilot0_lora.cga_trigger_fraction,
        "cga_warmup_steps": config.pilot0_lora.cga_warmup_steps,
        "cga_score_window": config.pilot0_lora.cga_score_window,
        "cga_improvement_margin": config.pilot0_lora.cga_improvement_margin,
        "cga_verified_commit": None,
        "cga_extra_batch_id": None,
        "cga_s1": first_probe_score,
        "cga_s2": None,
        "cga_score_improvement": None,
        "cga_q1_norm": first_q_norm,
        "cga_q2_norm": None,
        "cga_g2_norm": None,
        "ga2_recorded": False,
        "ga2_score": None,
        "ga2_delta_bar": None,
        "ga2_score_improvement": None,
        "ga2_q_norm": None,
        "ga2_g2_norm": None,
        "ga2_extra_backward_count": 0,
        "extra_backward_count": 0,
    }
    update_mode = config.pilot0_lora.update_mode
    online_ga2 = update_mode == "always_ga2"
    online_cga = update_mode in {"progate_cga", "progate_cga_verified"}
    online_verified = update_mode == "progate_cga_verified"
    record_ga2 = config.pilot0_lora.cga_record_always_ga2 or online_ga2
    if cga_mode == "off" and not record_ga2 and not online_cga:
        return result

    extra_batch = train_batches[(step + 1) % len(train_batches)]
    recent_scores = _recent_scores(score_history, config.pilot0_lora.cga_score_window)
    can_trigger = len(score_history) >= config.pilot0_lora.cga_warmup_steps and bool(recent_scores)
    threshold = _quantile(recent_scores, config.pilot0_lora.cga_trigger_fraction) if can_trigger else None
    if threshold is not None:
        result["cga_trigger_threshold"] = threshold
    triggered = bool(threshold is not None and first_probe_score <= threshold)
    needs_ga2 = record_ga2 or triggered
    leave_average_grads = online_ga2 or (online_cga and triggered)

    if needs_ga2:
        result.update(
            _measure_ga2_proposal(
                model=model,
                optimizer=optimizer,
                params=params,
                torch=torch,
                config=config,
                first_grads=first_grads,
                extra_batch=extra_batch,
                selected_probe_batches=selected_probe_batches,
                first_probe_score=first_probe_score,
                leave_average_grads=leave_average_grads,
            )
        )
        result["ga2_extra_backward_count"] = 1

    if online_ga2:
        result["cga_commit_source"] = "q2_always"
        result["extra_backward_count"] = 1
        return result

    if not triggered:
        return result

    if not result["ga2_recorded"]:
        result.update(
            _measure_ga2_proposal(
                model=model,
                optimizer=optimizer,
                params=params,
                torch=torch,
                config=config,
                first_grads=first_grads,
                extra_batch=extra_batch,
                selected_probe_batches=selected_probe_batches,
                first_probe_score=first_probe_score,
                leave_average_grads=leave_average_grads,
            )
        )
        result["ga2_extra_backward_count"] = 1

    improvement = result["ga2_score_improvement"]
    verified_commit = bool(not online_verified or (improvement is not None and improvement > config.pilot0_lora.cga_improvement_margin))
    if online_verified and not verified_commit:
        restore_grads(first_grads, params)

    result.update(
        {
            "cga_commit_source": _cga_commit_source(online_cga, online_verified, verified_commit),
            "cga_triggered": True,
            "cga_verified_commit": verified_commit if online_verified else None,
            "cga_extra_batch_id": extra_batch.batch_id,
            "cga_s2": result["ga2_score"],
            "cga_score_improvement": result["ga2_score_improvement"],
            "cga_q2_norm": result["ga2_q_norm"],
            "cga_g2_norm": result["ga2_g2_norm"],
            "extra_backward_count": 1,
        }
    )
    return result


def _cga_commit_source(online_cga: bool, online_verified: bool, verified_commit: bool) -> str:
    if not online_cga:
        return "q1"
    if not online_verified:
        return "q2_triggered"
    return "q2_verified" if verified_commit else "q1_verified_no_improve"


def _measure_ga2_proposal(
    model: Any,
    optimizer: Any,
    params: list[Any],
    torch: Any,
    config: LoraExperimentConfig,
    first_grads: list[Any | None],
    extra_batch: Batch,
    selected_probe_batches: list[Batch],
    first_probe_score: float,
    leave_average_grads: bool,
) -> dict[str, Any]:
    extra = _build_extra_evidence_proposal(
        model,
        optimizer,
        params,
        torch,
        config,
        first_grads,
        extra_batch,
        selected_probe_batches,
        first_probe_score,
        leave_average_grads,
    )
    return {
        "ga2_recorded": True,
        "ga2_score": extra["score"],
        "ga2_delta_bar": extra["delta_bar"],
        "ga2_score_improvement": extra["score_improvement"],
        "ga2_q_norm": extra["q_norm"],
        "ga2_g2_norm": extra["g2_norm"],
    }


def evaluate_loss(model: Any, batches: list[Batch]) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        import torch

        with torch.no_grad():
            for batch in batches:
                losses.append(_to_float(_loss(model, batch.tensors)))
    finally:
        model.train(was_training)
    return statistics.fmean(losses)


def apply_vector_to_params(params: list[Any], updates: list[Any], scale: float) -> None:
    for param, update in zip(params, updates, strict=True):
        param.data.add_(update, alpha=scale)


def set_params_to_scaled_proposal(params: list[Any], proposal: Proposal, scale: float) -> None:
    for param, before, update in zip(params, proposal.snapshot, proposal.updates, strict=True):
        param.data.copy_(before)
        param.data.add_(update, alpha=scale)


def restore_params(snapshot: list[Any], params: list[Any]) -> None:
    for before, param in zip(snapshot, params, strict=True):
        param.data.copy_(before)


def snapshot_params(params: list[Any]) -> list[Any]:
    return [param.detach().clone() for param in params]


def snapshot_grads(params: list[Any]) -> list[Any | None]:
    return [None if param.grad is None else param.grad.detach().clone() for param in params]


def restore_grads(snapshot: list[Any | None], params: list[Any]) -> None:
    for grad, param in zip(snapshot, params, strict=True):
        if grad is None:
            param.grad = None
        else:
            param.grad = grad.detach().clone()


def average_grads_with_snapshot(params: list[Any], first_grads: list[Any | None]) -> None:
    for param, first_grad in zip(params, first_grads, strict=True):
        second_grad = param.grad
        if first_grad is None and second_grad is None:
            param.grad = None
        elif first_grad is None:
            param.grad = 0.5 * second_grad.detach().clone()
        elif second_grad is None:
            param.grad = 0.5 * first_grad.detach().clone()
        else:
            param.grad = 0.5 * (first_grad.detach().clone() + second_grad.detach())


def grad_norm(params: list[Any], torch: Any) -> float:
    grads = [param.grad for param in params if param.grad is not None]
    return vector_norm(grads, torch) if grads else 0.0


def vector_norm(values: list[Any], torch: Any) -> float:
    total = 0.0
    for value in values:
        total += float(torch.sum(value.detach() * value.detach()).item())
    return math.sqrt(total)


def _loss(model: Any, tensors: dict[str, Any]) -> Any:
    output = model(**tensors)
    loss = getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError("model output did not include loss")
    return loss


def _make_batches(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    config: LoraExperimentConfig,
    batch_size: int,
    prefix: str,
    device: str,
) -> list[Batch]:
    encoded = [_encode_example(row, tokenizer, config.dataset) for row in rows]
    batches: list[Batch] = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        tensors = _collate(chunk, tokenizer.pad_token_id, device)
        batches.append(Batch(batch_id=f"{prefix}-{start // batch_size}", tensors=tensors))
    return batches


def _encode_example(row: dict[str, Any], tokenizer: Any, config: DatasetConfig) -> dict[str, list[int]]:
    instruction = str(row.get(config.instruction_field, "")).strip()
    extra_input = str(row.get(config.input_field, "")).strip()
    output = str(row.get(config.output_field, "")).strip()

    prompt_parts = ["### Instruction:", instruction]
    if extra_input:
        prompt_parts.extend(["", "### Input:", extra_input])
    prompt_parts.extend(["", "### Response:"])
    prompt = "\n".join(prompt_parts) + "\n"
    completion = output + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        prompt + completion,
        add_special_tokens=False,
        max_length=config.max_length,
        truncation=True,
    )
    input_ids = list(encoded["input_ids"])
    labels = input_ids.copy()
    prefix_len = min(len(prompt_ids), len(labels))
    labels[:prefix_len] = [-100] * prefix_len
    if labels and all(value == -100 for value in labels):
        labels[-1] = input_ids[-1]
    return {"input_ids": input_ids, "labels": labels}


def _collate(features: list[dict[str, list[int]]], pad_token_id: int, device: str) -> dict[str, Any]:
    import torch

    max_length = max(len(feature["input_ids"]) for feature in features)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for feature in features:
        pad = max_length - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * pad)
        attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad)
        labels.append(feature["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _split_rows(dataset: Any, config: LoraExperimentConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_end = config.dataset.train_size
    probe_end = train_end + config.dataset.probe_size
    validation_end = probe_end + config.dataset.validation_size
    rows = [dict(dataset[index]) for index in range(validation_end)]
    return rows[:train_end], rows[train_end:probe_end], rows[probe_end:validation_end]


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(torch: Any, requested: str, device: str) -> Any | None:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"unknown dtype: {requested}")
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return None


def _load_model(transformers: Any, config: ModelConfig, load_kwargs: dict[str, Any]) -> tuple[Any, str]:
    available = {
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoModelForImageTextToText": getattr(transformers, "AutoModelForImageTextToText", None),
    }
    if config.model_class == "auto-image-text":
        order = ["AutoModelForImageTextToText", "AutoModelForCausalLM"]
    elif config.model_class == "image-text":
        order = ["AutoModelForImageTextToText"]
    elif config.model_class == "causal-lm":
        order = ["AutoModelForCausalLM"]
    elif config.model_class == "auto":
        order = ["AutoModelForCausalLM", "AutoModelForImageTextToText"]
    else:
        raise ValueError(f"unknown model_class: {config.model_class}")

    errors: list[dict[str, str]] = []
    for class_name in order:
        loader = available.get(class_name)
        if loader is None:
            errors.append({"class": class_name, "error": "class unavailable"})
            continue
        try:
            return loader.from_pretrained(config.name, **load_kwargs), class_name
        except Exception as error:  # noqa: BLE001 - try the configured fallback.
            errors.append({"class": class_name, "error": repr(error)})
    raise RuntimeError(f"failed to load model with configured policy {config.model_class}: {errors}")


def _lora_exclude_modules(config: LoraConfig, model_load_class: str) -> str | None:
    if config.exclude_modules is not None:
        return config.exclude_modules
    if model_load_class == "AutoModelForImageTextToText":
        return r".*(vision_tower|audio_tower|embed_vision|embed_audio).*"
    return None


def _effective_device_map(config: ModelConfig) -> str | None:
    if not config.device_map or config.device_map == "none":
        return None
    return config.device_map


def _uses_device_map(config: ModelConfig) -> bool:
    return _effective_device_map(config) is not None


def _summary(config: LoraExperimentConfig, rows: list[dict[str, Any]], trainable: list[Any], device: str) -> dict[str, Any]:
    scored = [row for row in rows if row["validation_loss_snapshot"] is not None]
    horizon_rows = [row for row in rows if row.get("T_e_active")]
    alpha_values = [row.get("alpha_t") for row in rows if row.get("alpha_t") is not None]
    horizon_alpha_values = [row.get("alpha_t") for row in horizon_rows if row.get("alpha_t") is not None]
    early_validation = [row["validation_loss_snapshot"] for row in horizon_rows if row["validation_loss_snapshot"] is not None]
    cga_rows = [row for row in rows if row.get("cga_triggered")]
    verified_commits = [row for row in cga_rows if row.get("cga_verified_commit") is True]
    cga_improvements = [row.get("cga_score_improvement") for row in cga_rows]
    ga2_rows = [row for row in rows if row.get("ga2_recorded")]
    ga2_improvements = [row.get("ga2_score_improvement") for row in ga2_rows]
    cga_extra_backward_count = sum(int(row.get("extra_backward_count") or 0) for row in rows)
    ga2_extra_backward_count = sum(int(row.get("ga2_extra_backward_count") or 0) for row in rows)
    total_backward_count = sum(int(row.get("backward_count_total") or 1) for row in rows)
    control_extra_backward_count = sum(int(row.get("backward_count_extra") or 0) for row in rows)
    control_triggered_rows = [row for row in rows if row.get("control_triggered")]
    control_score_improvements = [row.get("score_improvement") for row in control_triggered_rows]
    commit_alphas = [row.get("commit_alpha") for row in rows]
    effective_update_norms = [row.get("effective_q_norm") for row in rows]
    return {
        "mode": "real_lora",
        "update_mode": config.pilot0_lora.update_mode,
        "control_mode": config.pilot0_lora.control_mode,
        "model_name": config.model.name,
        "dataset_name": config.dataset.name,
        "rows": len(rows),
        "validation_snapshots": len(scored),
        "device": device,
        "lora_rank": config.lora.rank,
        "trainable_parameter_count": sum(param.numel() for param in trainable),
        "mean_probe_score": _mean([row["probe_score"] for row in rows]),
        "mean_delta_bar": _mean([row["delta_bar"] for row in rows]),
        "mean_train_loss": _mean([row["train_loss"] for row in rows]),
        "mean_validation_loss": _mean([row["validation_loss_snapshot"] for row in scored]),
        "early_validation_loss_mean": _mean(early_validation),
        "final_validation_loss": scored[-1]["validation_loss_snapshot"] if scored else None,
        "loss_spikes": sum(1 for row in rows if row["loss_spike"]),
        "horizon_steps": _verification_horizon_steps(config.pilot0_lora),
        "active_horizon_steps": len(horizon_rows),
        "mean_alpha": _mean(alpha_values),
        "mean_alpha_during_horizon": _mean(horizon_alpha_values),
        "min_alpha": min(alpha_values) if alpha_values else None,
        "max_alpha": max(alpha_values) if alpha_values else None,
        "fraction_alpha_below_0_5": _fraction(alpha_values, lambda value: value < 0.5),
        "fraction_alpha_above_0_8": _fraction(alpha_values, lambda value: value > 0.8),
        "fraction_horizon_alpha_below_0_5": _fraction(horizon_alpha_values, lambda value: value < 0.5),
        "fraction_horizon_alpha_above_0_8": _fraction(horizon_alpha_values, lambda value: value > 0.8),
        "cga_mode": config.pilot0_lora.cga_mode,
        "cga_triggered_steps": len(cga_rows),
        "cga_trigger_rate": len(cga_rows) / len(rows) if rows else None,
        "cga_extra_backward_count": cga_extra_backward_count,
        "cga_extra_backward_overhead": cga_extra_backward_count / len(rows) if rows else None,
        "cga_verified_commits": len(verified_commits),
        "cga_verified_commit_rate": len(verified_commits) / len(cga_rows) if cga_rows else None,
        "cga_mean_score_improvement": _mean(cga_improvements),
        "cga_median_score_improvement": _median(cga_improvements),
        "cga_fraction_score_improved": _fraction(cga_improvements, lambda value: value > 0.0),
        "ga2_recorded_steps": len(ga2_rows),
        "ga2_extra_backward_count": ga2_extra_backward_count,
        "ga2_extra_backward_overhead": ga2_extra_backward_count / len(rows) if rows else None,
        "ga2_mean_score_improvement": _mean(ga2_improvements),
        "ga2_median_score_improvement": _median(ga2_improvements),
        "ga2_fraction_score_improved": _fraction(ga2_improvements, lambda value: value > 0.0),
        "total_backward_count": total_backward_count,
        "extra_backward_count": control_extra_backward_count,
        "extra_backward_rate": control_extra_backward_count / len(rows) if rows else None,
        "control_triggered_steps": len(control_triggered_rows),
        "control_trigger_rate": len(control_triggered_rows) / len(rows) if rows else None,
        "mean_score_improvement_on_triggered": _mean(control_score_improvements),
        "fraction_score_improved_on_triggered": _fraction(control_score_improvements, lambda value: value > 0.0),
        "mean_commit_alpha": _mean(commit_alphas),
        "effective_update_norm_mean": _mean(effective_update_norms),
        "future_windows": config.pilot0_lora.future_windows,
    }


def _write_run_note(run_dir: Path, config: LoraExperimentConfig, summary: dict[str, Any]) -> None:
    lines = [
        f"# {config.run.name}",
        "",
        "## Purpose",
        "",
        _purpose_text(config),
        "",
        "## Model",
        "",
        config.model.name,
        "",
        "## Dataset",
        "",
        config.dataset.name,
        "",
        "## Summary",
        "",
        f"- rows: {summary['rows']}",
        f"- validation snapshots: {summary['validation_snapshots']}",
        f"- trainable parameters: {summary['trainable_parameter_count']}",
        f"- mean probe score: {summary['mean_probe_score']}",
        f"- mean validation loss: {summary['mean_validation_loss']}",
        f"- final validation loss: {summary['final_validation_loss']}",
        f"- mean alpha during horizon: {summary['mean_alpha_during_horizon']}",
        f"- control mode: {summary['control_mode']}",
        f"- total backward count: {summary['total_backward_count']}",
        f"- extra backward rate: {summary['extra_backward_rate']}",
        f"- control trigger rate: {summary['control_trigger_rate']}",
        f"- cga triggered steps: {summary['cga_triggered_steps']}",
        f"- cga mean score improvement: {summary['cga_mean_score_improvement']}",
        "",
        "## Reminder",
        "",
        _reminder_text(config),
        "",
    ]
    (run_dir / "run_note.md").write_text("\n".join(lines), encoding="utf-8")


def _write_interrupted(run_dir: Path, reason: str) -> None:
    metrics_path = run_dir / "metrics.jsonl"
    write_json(
        run_dir / "interrupted.json",
        {
            "status": "interrupted",
            "reason": reason,
            "metrics_rows": _line_count(metrics_path),
        },
    )


def _write_interrupted_run_note(run_dir: Path, config: LoraExperimentConfig) -> None:
    lines = [
        f"# {config.run.name}",
        "",
        "## Status",
        "",
        "Interrupted before completion.",
        "",
        "## Reminder",
        "",
        "Use `interrupted.json` and partial `metrics.jsonl` only as execution evidence, not as experiment results.",
        "",
    ]
    (run_dir / "run_note.md").write_text("\n".join(lines), encoding="utf-8")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _required(values: dict[str, Any], key: str) -> Any:
    if key not in values:
        raise KeyError(f"missing required config key: {key}")
    return values[key]


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _verification_horizon_steps(config: Pilot0LoraConfig) -> int:
    if config.update_mode != "progate_lite":
        return 0
    return max(1, int(math.ceil(config.steps * config.verification_horizon_fraction)))


def _effective_alpha(alpha_probe: float, horizon_active: bool, alpha_min: float) -> float:
    if not horizon_active:
        return 1.0
    return alpha_min + (1.0 - alpha_min) * alpha_probe


def _to_float(value: Any) -> float:
    return float(value.detach().cpu().item())


def _validate_pilot0_lora_config(config: Pilot0LoraConfig) -> None:
    if config.update_mode not in {"adamw", "progate_lite", "always_ga2", "progate_cga", "progate_cga_verified"}:
        raise ValueError(f"unknown update_mode: {config.update_mode}")
    if config.control_mode not in {"off", "scsa_damp", "scsa_extra", "random_extra", "fixed_extra"}:
        raise ValueError(f"unknown control_mode: {config.control_mode}")
    if config.control_mode != "off" and config.update_mode != "adamw":
        raise ValueError("SCSA control modes require update_mode='adamw'")
    if config.cga_mode not in {"off", "diagnostic", "online_update"}:
        raise ValueError(f"unknown cga_mode: {config.cga_mode}")
    if config.control_mode != "off" and config.cga_mode != "off":
        raise ValueError("SCSA control modes cannot be combined with legacy cga_mode")
    if config.cga_mode == "diagnostic" and config.update_mode != "adamw":
        raise ValueError("cga diagnostic must keep update_mode='adamw'")
    if config.cga_mode == "online_update" and config.update_mode not in {"always_ga2", "progate_cga", "progate_cga_verified"}:
        raise ValueError("cga online_update must use update_mode='always_ga2', 'progate_cga', or 'progate_cga_verified'")
    if config.update_mode in {"always_ga2", "progate_cga", "progate_cga_verified"} and config.cga_mode != "online_update":
        raise ValueError("always_ga2/progate_cga/progate_cga_verified require cga_mode='online_update'")
    if not 0.0 < config.cga_trigger_fraction < 1.0:
        raise ValueError("cga_trigger_fraction must be between 0 and 1")
    if config.cga_warmup_steps < 0:
        raise ValueError("cga_warmup_steps must be non-negative")
    if config.cga_score_window < 0:
        raise ValueError("cga_score_window must be non-negative")
    if not 0.0 < config.control_trigger_fraction < 1.0:
        raise ValueError("control_trigger_fraction must be between 0 and 1")
    if config.control_warmup_steps < 0:
        raise ValueError("control_warmup_steps must be non-negative")
    if config.control_score_window < 0:
        raise ValueError("control_score_window must be non-negative")
    if not 0.0 < config.control_damp_factor <= 1.0:
        raise ValueError("control_damp_factor must be in (0, 1]")
    if config.control_fixed_period <= 0:
        raise ValueError("control_fixed_period must be positive")
    if config.control_extra_policy != "average_grad":
        raise ValueError("only control_extra_policy='average_grad' is currently supported")


def _recent_scores(score_history: list[float], window: int) -> list[float]:
    if window <= 0:
        return list(score_history)
    return score_history[-window:]


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = int(math.floor((len(ordered) - 1) * fraction))
    return ordered[index]


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _median(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.median(clean) if clean else None


def _fraction(values: list[float | None], predicate: Any) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(1 for value in clean if predicate(value)) / len(clean)


def _purpose_text(config: LoraExperimentConfig) -> str:
    if config.pilot0_lora.control_mode == "scsa_damp":
        return "Run SCSA-Damp. Bottom-quantile ProGate proposals commit a damped q1 displacement under unchanged AdamW moment updates."
    if config.pilot0_lora.control_mode == "scsa_extra":
        return "Run SCSA-Extra. Bottom-quantile ProGate proposals receive one extra microbatch gradient and commit the averaged-gradient q2 proposal."
    if config.pilot0_lora.control_mode == "random_extra":
        return "Run a random-extra baseline. Extra evidence is allocated at the configured random trigger rate."
    if config.pilot0_lora.control_mode == "fixed_extra":
        return "Run a fixed-period-extra baseline. Extra evidence is allocated at the configured step period."
    if config.pilot0_lora.update_mode == "progate_cga_verified":
        return "Run an online ProGate-CGA-Verified update smoke. Low-score proposals compute q2, but commit q2 only when the second probe improves over q1."
    if config.pilot0_lora.update_mode == "progate_cga":
        return "Run an online ProGate-CGA update smoke. Low-score proposals receive one extra microbatch and commit the averaged-gradient AdamW proposal."
    if config.pilot0_lora.update_mode == "always_ga2":
        return "Run an Always-GA(2) update smoke. Every step commits an AdamW proposal from two averaged microbatch gradients."
    if config.pilot0_lora.cga_mode == "diagnostic":
        return "Run ProGate-CGA diagnostic. AdamW updates are unchanged; low-score proposals receive an extra microbatch proposal for measurement only."
    if config.pilot0_lora.update_mode == "progate_lite":
        return "Run a small ProGate-Lite update smoke. The core soft probe gate is applied during the early horizon; guards remain disabled."
    return "Run logging-only real Pilot 0. AdamW updates are unchanged; ProGate signals are recorded only."


def _reminder_text(config: LoraExperimentConfig) -> str:
    if config.pilot0_lora.control_mode != "off":
        return "This is an SCSA control run. Interpret it only against same-backward-budget baselines and the logged cumulative backward count."
    if config.pilot0_lora.update_mode in {"progate_cga", "always_ga2"}:
        return "This is an online update smoke. Interpret it as stability and trajectory evidence only after comparing against AdamW under the same budget."
    if config.pilot0_lora.cga_mode == "diagnostic":
        return "This is not a CGA update result. It only tests whether extra microbatch evidence improves low-score proposals."
    if config.pilot0_lora.update_mode == "progate_lite":
        return "This is an update smoke, not a full method result. It only tests whether the core gate can touch parameter movement without destabilizing training."
    return "This run must not be interpreted as ProGate performance. It only tests whether probe score predicts future validation delta."
