from __future__ import annotations

from pathlib import Path
import argparse

from .analyze_gate import analyze_gate_behavior
from .analyze_cga import analyze_cga_diagnostic
from .analyze_pilot0 import analyze_run
from .analyze_verified_cga import analyze_verified_cga
from .aggressive_configs import generate_aggressive_configs
from .benchmark_overhead import run_overhead_benchmark
from .bootstrap_gap import bootstrap_top_bottom_gap
from .compare_updates import compare_update_runs
from .config import load_config
from .control_analysis import analyze_control_runs
from .diagnostic_baselines import analyze_diagnostic_baselines
from .pilot0 import run_pilot0
from .pilot0_lora import load_lora_config, run_pilot0_lora
from .robustness import analyze_proposal_predictability
from .smoke_model import DEFAULT_MODEL, DTYPES, MODEL_CLASSES, SMOKE_STAGES, SmokeModelConfig, run_smoke_model
from .summarize_sweep import summarize_sweep


def main() -> None:
    parser = argparse.ArgumentParser(prog="progate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot0 = subparsers.add_parser("pilot0", help="run the Pilot 0 logging sanity experiment")
    pilot0.add_argument("--config", type=Path, required=True)
    pilot0.add_argument("--project-root", type=Path, default=Path.cwd())

    pilot0_lora = subparsers.add_parser("pilot0-lora", help="run logging-only real LoRA Pilot 0")
    pilot0_lora.add_argument("--config", type=Path, required=True)
    pilot0_lora.add_argument("--project-root", type=Path, default=Path.cwd())

    overhead = subparsers.add_parser("benchmark-overhead", help="benchmark ProGate diagnostic step-time overhead")
    overhead.add_argument("--config", type=Path, required=True)
    overhead.add_argument("--project-root", type=Path, default=Path.cwd())

    analyze = subparsers.add_parser("analyze-pilot0", help="analyze a Pilot 0 metrics.jsonl run")
    analyze.add_argument("--run-dir", type=Path, required=True)
    analyze.add_argument("--bins", type=int, default=5)
    analyze.add_argument("--alpha-acc", type=float, default=0.8)
    analyze.add_argument("--alpha-rej", type=float, default=0.1)
    analyze.add_argument("--top-fraction", type=float, default=0.2)
    analyze.add_argument("--future-windows", default="1,5,10,20")

    smoke = subparsers.add_parser("smoke-model", help="smoke-test the fixed Pilot 0 model path")
    smoke.add_argument("--model", default=DEFAULT_MODEL)
    smoke.add_argument("--stage", choices=SMOKE_STAGES, default="proposal")
    smoke.add_argument("--seed", type=int, default=20260511)
    smoke.add_argument("--output-dir", type=Path, default=Path("runs"))
    smoke.add_argument("--project-root", type=Path, default=Path.cwd())
    smoke.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    smoke.add_argument("--model-class", default="auto", choices=MODEL_CLASSES)
    smoke.add_argument("--dtype", default="auto", choices=DTYPES)
    smoke.add_argument("--device-map", default="none")
    smoke.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=False)
    smoke.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    smoke.add_argument("--local-files-only", action="store_true")
    smoke.add_argument("--max-length", type=int, default=32)
    smoke.add_argument("--probe-k", type=int, default=1)
    smoke.add_argument("--max-linear-modules", type=int, default=200)

    sweep = subparsers.add_parser("summarize-sweep", help="summarize analyzed Pilot 0 runs")
    sweep.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    sweep.add_argument("--output-dir", type=Path, required=True)

    bootstrap = subparsers.add_parser("bootstrap-gap", help="bootstrap top-vs-bottom probe score gaps")
    bootstrap.add_argument("--run-dir", type=Path, required=True)
    bootstrap.add_argument("--future-windows", default="1,5,10,20")
    bootstrap.add_argument("--top-fraction", type=float, default=0.2)
    bootstrap.add_argument("--bootstrap-n", type=int, default=5000)
    bootstrap.add_argument("--seed", type=int, default=20260511)

    robustness = subparsers.add_parser(
        "proposal-predictability",
        help="compute rank, AUC, and permutation robustness for proposal scores",
    )
    robustness.add_argument("--run-dir", type=Path, required=True)
    robustness.add_argument("--output-dir", type=Path)
    robustness.add_argument("--future-windows", default="1,5,10,20")
    robustness.add_argument("--top-fractions", default="0.1,0.2,0.3")
    robustness.add_argument("--permutation-n", type=int, default=5000)
    robustness.add_argument("--seed", type=int, default=20260511)

    baselines = subparsers.add_parser(
        "diagnostic-baselines",
        help="compare ProGate score against simple diagnostic proxy scores",
    )
    baselines.add_argument("--run-dir", type=Path, required=True)
    baselines.add_argument("--output-dir", type=Path)
    baselines.add_argument("--future-windows", default="1,5,10,20")
    baselines.add_argument("--top-fraction", type=float, default=0.2)
    baselines.add_argument("--permutation-n", type=int, default=5000)
    baselines.add_argument("--seed", type=int, default=20260511)

    compare = subparsers.add_parser("compare-updates", help="compare AdamW and ProGate update runs")
    compare.add_argument("--baseline-run-dirs", type=Path, nargs="+", required=True)
    compare.add_argument("--candidate-run-dirs", type=Path, nargs="+", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--early-steps", type=int, default=6)

    gate = subparsers.add_parser("analyze-gate", help="analyze alpha/probe behavior in ProGate update runs")
    gate.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    gate.add_argument("--output-dir", type=Path, required=True)
    gate.add_argument("--future-windows", default="1,5,10,20")
    gate.add_argument("--low-alpha", type=float, default=0.5)
    gate.add_argument("--high-alpha", type=float, default=0.8)

    cga = subparsers.add_parser("analyze-cga", help="analyze ProGate-CGA diagnostic runs")
    cga.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    cga.add_argument("--output-dir", type=Path, required=True)

    verified_cga = subparsers.add_parser("analyze-verified-cga", help="analyze Verified-CGA decision behavior")
    verified_cga.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    verified_cga.add_argument("--output-dir", type=Path, required=True)
    verified_cga.add_argument("--future-windows", default="5,10,20")

    aggressive_configs = subparsers.add_parser("generate-aggressive-configs", help="generate aggressive redesign configs")
    aggressive_configs.add_argument("--stage", choices=["stage1a", "stage2"], required=True)
    aggressive_configs.add_argument("--output-dir", type=Path, required=True)
    aggressive_configs.add_argument("--winner-k", type=int)
    aggressive_configs.add_argument("--validation-size", type=int, default=512)

    control_analysis = subparsers.add_parser("analyze-control", help="summarize SCSA/control runs by method and budget")
    control_analysis.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    control_analysis.add_argument("--output-dir", type=Path, required=True)
    control_analysis.add_argument("--bad-run-threshold", type=float)

    args = parser.parse_args()
    if args.command == "pilot0":
        config_path = args.config.resolve()
        project_root = args.project_root.resolve()
        config = load_config(config_path)
        run_dir = run_pilot0(config, config_path, project_root)
        print(f"run_dir={run_dir}")
    elif args.command == "pilot0-lora":
        config_path = args.config.resolve()
        project_root = args.project_root.resolve()
        config = load_lora_config(config_path)
        run_dir = run_pilot0_lora(config, config_path, project_root)
        print(f"run_dir={run_dir}")
    elif args.command == "benchmark-overhead":
        config_path = args.config.resolve()
        project_root = args.project_root.resolve()
        config = load_lora_config(config_path)
        run_dir = run_overhead_benchmark(config, config_path, project_root)
        print(f"run_dir={run_dir}")
    elif args.command == "analyze-pilot0":
        analysis_dir = analyze_run(
            args.run_dir.resolve(),
            bins=args.bins,
            alpha_acc=args.alpha_acc,
            alpha_rej=args.alpha_rej,
            top_fraction=args.top_fraction,
            future_windows=_parse_windows(args.future_windows),
        )
        print(f"analysis_dir={analysis_dir}")
    elif args.command == "smoke-model":
        run_dir = run_smoke_model(
            SmokeModelConfig(
                model=args.model,
                seed=args.seed,
                output_dir=args.output_dir.resolve(),
                project_root=args.project_root.resolve(),
                stage=args.stage,
                device=args.device,
                model_class=args.model_class,
                dtype=args.dtype,
                device_map=args.device_map,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                max_length=args.max_length,
                probe_k=args.probe_k,
                max_linear_modules=args.max_linear_modules,
            )
        )
        print(f"run_dir={run_dir}")
    elif args.command == "summarize-sweep":
        output_dir = summarize_sweep(
            [path.resolve() for path in args.run_dirs],
            args.output_dir.resolve(),
        )
        print(f"summary_dir={output_dir}")
    elif args.command == "bootstrap-gap":
        analysis_dir = bootstrap_top_bottom_gap(
            args.run_dir.resolve(),
            future_windows=_parse_windows(args.future_windows),
            top_fraction=args.top_fraction,
            bootstrap_n=args.bootstrap_n,
            seed=args.seed,
        )
        print(f"analysis_dir={analysis_dir}")
    elif args.command == "proposal-predictability":
        output_dir = analyze_proposal_predictability(
            args.run_dir.resolve(),
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            future_windows=_parse_windows(args.future_windows),
            top_fractions=_parse_floats(args.top_fractions),
            permutation_n=args.permutation_n,
            seed=args.seed,
        )
        print(f"analysis_dir={output_dir}")
    elif args.command == "diagnostic-baselines":
        output_dir = analyze_diagnostic_baselines(
            args.run_dir.resolve(),
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            future_windows=_parse_windows(args.future_windows),
            top_fraction=args.top_fraction,
            permutation_n=args.permutation_n,
            seed=args.seed,
        )
        print(f"analysis_dir={output_dir}")
    elif args.command == "compare-updates":
        output_dir = compare_update_runs(
            [path.resolve() for path in args.baseline_run_dirs],
            [path.resolve() for path in args.candidate_run_dirs],
            args.output_dir.resolve(),
            early_steps=args.early_steps,
        )
        print(f"comparison_dir={output_dir}")
    elif args.command == "analyze-gate":
        output_dir = analyze_gate_behavior(
            [path.resolve() for path in args.run_dirs],
            args.output_dir.resolve(),
            future_windows=_parse_windows(args.future_windows),
            low_alpha=args.low_alpha,
            high_alpha=args.high_alpha,
        )
        print(f"analysis_dir={output_dir}")
    elif args.command == "analyze-cga":
        output_dir = analyze_cga_diagnostic(
            [path.resolve() for path in args.run_dirs],
            args.output_dir.resolve(),
        )
        print(f"analysis_dir={output_dir}")
    elif args.command == "analyze-verified-cga":
        output_dir = analyze_verified_cga(
            [path.resolve() for path in args.run_dirs],
            args.output_dir.resolve(),
            future_windows=_parse_windows(args.future_windows),
        )
        print(f"analysis_dir={output_dir}")
    elif args.command == "generate-aggressive-configs":
        paths = generate_aggressive_configs(
            args.stage,
            args.output_dir.resolve(),
            winner_k=args.winner_k,
            validation_size=args.validation_size,
        )
        print(f"generated={len(paths)}")
        for path in paths:
            print(path)
    elif args.command == "analyze-control":
        output_dir = analyze_control_runs(
            [path.resolve() for path in args.run_dirs],
            args.output_dir.resolve(),
            bad_run_threshold=args.bad_run_threshold,
        )
        print(f"analysis_dir={output_dir}")


def _parse_windows(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]
