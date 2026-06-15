from __future__ import annotations

from pathlib import Path
import math
import textwrap


SEEDS = [20260511, 20260512, 20260513, 20260514, 20260515]


def generate_aggressive_configs(
    stage: str,
    output_dir: Path,
    winner_k: int | None = None,
    validation_size: int = 512,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == "stage1a":
        return _generate_stage1a(output_dir)
    if stage == "stage2":
        if winner_k is None:
            raise ValueError("--winner-k is required for stage2")
        return _generate_stage2(output_dir, winner_k=winner_k, validation_size=validation_size)
    raise ValueError(f"unknown aggressive config stage: {stage}")


def _generate_stage1a(output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for probe_k in [2, 4, 8]:
        for seed in SEEDS:
            name = f"stage1a-qwen35-2b-alpaca-val512-k{probe_k}"
            path = output_dir / f"stage1a_qwen35_2b_alpaca_val512_k{probe_k}_seed{seed}.toml"
            path.write_text(
                _config_text(
                    run_name=name,
                    seed=seed,
                    model="Qwen/Qwen3.5-2B-Base",
                    train_size=512,
                    probe_size=128,
                    validation_size=512,
                    validation_batch_size=8,
                    steps=300,
                    probe_k=probe_k,
                    control_mode="off",
                ),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def _generate_stage2(output_dir: Path, winner_k: int, validation_size: int) -> list[Path]:
    paths: list[Path] = []
    base_steps = 500
    methods = [
        ("adamw", "off", base_steps, {}),
        ("scsa-damp", "scsa_damp", base_steps, {"control_damp_factor": "0.5"}),
        ("scsa-extra", "scsa_extra", base_steps, {}),
        ("random-extra", "random_extra", base_steps, {}),
        ("fixed-extra", "fixed_extra", base_steps, {}),
        ("adamw-longer-1p25", "off", math.ceil(base_steps * 1.25), {}),
    ]
    for method_name, control_mode, steps, extras in methods:
        for seed in SEEDS:
            name = f"stage2-qwen35-2b-alpaca-val{validation_size}-k{winner_k}-{method_name}"
            path = (
                output_dir
                / f"stage2_qwen35_2b_alpaca_val{validation_size}_k{winner_k}_{method_name.replace('-', '_')}_seed{seed}.toml"
            )
            path.write_text(
                _config_text(
                    run_name=name,
                    seed=seed,
                    model="Qwen/Qwen3.5-2B-Base",
                    train_size=512,
                    probe_size=128,
                    validation_size=validation_size,
                    validation_batch_size=8,
                    steps=steps,
                    probe_k=winner_k,
                    control_mode=control_mode,
                    extra_pilot0=extras,
                ),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def _config_text(
    run_name: str,
    seed: int,
    model: str,
    train_size: int,
    probe_size: int,
    validation_size: int,
    validation_batch_size: int,
    steps: int,
    probe_k: int,
    control_mode: str,
    extra_pilot0: dict[str, str] | None = None,
) -> str:
    pilot0_extras = {
        "control_trigger_fraction": "0.25",
        "control_warmup_steps": "30",
        "control_score_window": "50",
        "control_damp_factor": "0.5",
        "control_fixed_period": "4",
        "control_verify_improvement": "false",
        "control_improvement_margin": "0.0",
        "control_extra_policy": '"average_grad"',
    }
    pilot0_extras.update(extra_pilot0 or {})
    extra_lines = "\n".join(f"{key} = {value}" for key, value in pilot0_extras.items())
    return textwrap.dedent(
        f"""
        [run]
        name = "{run_name}"
        seed = {seed}
        output_dir = "runs"

        [model]
        name = "{model}"
        trust_remote_code = true
        device = "auto"
        model_class = "causal-lm"
        dtype = "bfloat16"
        device_map = "none"
        low_cpu_mem_usage = true
        local_files_only = false

        [dataset]
        name = "yahma/alpaca-cleaned"
        split = "train"
        train_size = {train_size}
        probe_size = {probe_size}
        validation_size = {validation_size}
        max_length = 256
        train_batch_size = 1
        probe_batch_size = 1
        validation_batch_size = {validation_batch_size}
        instruction_field = "instruction"
        input_field = "input"
        output_field = "output"

        [lora]
        rank = 8
        alpha = 16
        dropout = 0.0
        target_modules = [
          "q_proj",
          "k_proj",
          "v_proj",
          "o_proj",
          "gate_proj",
          "up_proj",
          "down_proj",
        ]

        [optimizer]
        lr = 1e-4
        weight_decay = 0.0
        betas = [0.9, 0.999]
        eps = 1e-8

        [pilot0_lora]
        steps = {steps}
        lambda_v = 1.0
        probe_k = {probe_k}
        tau_delta = 0.0
        gamma = 0.01
        validation_interval = 5
        loss_spike_threshold = 0.5
        future_windows = [1, 5, 10, 20]
        update_mode = "adamw"
        control_mode = "{control_mode}"
        verification_horizon_fraction = 0.0
        alpha_min = 0.0
        accepted_like_threshold = 0.8
        cga_mode = "off"
        cga_trigger_fraction = 0.3
        cga_warmup_steps = 20
        cga_score_window = 50
        cga_record_always_ga2 = false
        cga_improvement_margin = 0.0
        {extra_lines}
        """
    ).lstrip()
