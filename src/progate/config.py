from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class Pilot0Config:
    steps: int
    dimension: int
    learning_rate: float
    lambda_v: float
    probe_k: int
    future_window: int
    train_noise: float
    probe_noise: float
    validation_noise: float
    tau_delta: float
    gamma: float
    loss_spike_threshold: float


@dataclass(frozen=True)
class AnalysisConfig:
    bins: int
    alpha_acc: float
    alpha_rej: float


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    pilot0: Pilot0Config
    analysis: AnalysisConfig


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    run = raw.get("run", {})
    pilot0 = raw.get("pilot0", {})
    analysis = raw.get("analysis", {})

    return ExperimentConfig(
        run=RunConfig(
            name=str(_required(run, "name")),
            seed=int(_required(run, "seed")),
            output_dir=Path(str(_required(run, "output_dir"))),
        ),
        pilot0=Pilot0Config(
            steps=int(_required(pilot0, "steps")),
            dimension=int(_required(pilot0, "dimension")),
            learning_rate=float(_required(pilot0, "learning_rate")),
            lambda_v=float(_required(pilot0, "lambda_v")),
            probe_k=int(_required(pilot0, "probe_k")),
            future_window=int(_required(pilot0, "future_window")),
            train_noise=float(_required(pilot0, "train_noise")),
            probe_noise=float(_required(pilot0, "probe_noise")),
            validation_noise=float(_required(pilot0, "validation_noise")),
            tau_delta=float(_required(pilot0, "tau_delta")),
            gamma=float(_required(pilot0, "gamma")),
            loss_spike_threshold=float(_required(pilot0, "loss_spike_threshold")),
        ),
        analysis=AnalysisConfig(
            bins=int(_required(analysis, "bins")),
            alpha_acc=float(_required(analysis, "alpha_acc")),
            alpha_rej=float(_required(analysis, "alpha_rej")),
        ),
    )


def _required(values: dict[str, Any], key: str) -> Any:
    if key not in values:
        raise KeyError(f"missing required config key: {key}")
    return values[key]
