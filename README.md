# Forward-Only Proposal Diagnostics

This repository studies whether an optimizer's proposed update can be inspected
before it is applied during LoRA adaptation of language models.

At training step `t`, AdamW proposes a signed LoRA parameter displacement
`q_t`. This project temporarily applies `q_t` to the trainable LoRA parameters,
measures loss change on separate diagnostic batches using forward passes only,
restores the original parameters, and then applies the normal AdamW update.

The current public repository is intentionally minimal. It contains the core
code and a small public reproducibility bundle, while larger internal experiment
logs, private notes, and draft manuscripts are not included.

## Core Idea

```text
1. Compute the usual training gradient on the training batch.
2. Build AdamW's signed proposal q_t.
3. Measure diagnostic loss before applying q_t.
4. Temporarily apply q_t to the LoRA parameters.
5. Measure diagnostic loss after applying q_t.
6. Restore the original parameters.
7. Apply the normal AdamW update.
```

The diagnostic score is based on the loss decrease induced by the virtual
proposal. Higher score means the proposed update reduced diagnostic loss more.

The important separation is:

```text
training data: creates the AdamW proposal
diagnostic/probe data: scores the proposal before application
validation data: measures future validation loss movement for analysis
```

## Public Repository Contents

```text
src/progate/                     core implementation
configs/public/                  runnable public configs
docs/results/stage1a_summary.md  compact result summary
README.md
pyproject.toml
uv.lock
LICENSE
```

The public configs are:

```text
configs/public/qwen35_08b_diagnostic_smoke.toml
configs/public/qwen35_2b_stage1a_k2_seed20260511.toml
```

The first config is a small smoke test. The second is one Stage 1a diagnostic
run matching the public result setting.

## Setup

This project is tested with `uv`.

```bash
uv sync
```

The lockfile currently targets the development environment used for the
experiments. If `uv sync` cannot resolve your local Python installation, install
the Python version requested by `pyproject.toml` through `uv`, then run `uv sync`
again.

```bash
uv python install 3.14.5
uv sync
```

## Quick Smoke Test

This only checks that model loading, LoRA attachment, AdamW proposal extraction,
and the forward-only virtual probe path work.

```bash
uv run progate smoke-model \
  --model Qwen/Qwen3.5-0.8B-Base \
  --stage proposal \
  --model-class causal-lm \
  --dtype bfloat16
```

## Run a Small Public Diagnostic Smoke

This runs 5 LoRA steps on a tiny split. It is meant to verify the training and
logging path, not to reproduce the paper-scale result.

```bash
uv run progate pilot0-lora \
  --config configs/public/qwen35_08b_diagnostic_smoke.toml
```

Each run writes:

```text
runs/<run-id>/
  config.toml
  env.json
  metrics.jsonl
  summary.json
  run_note.md
```

## Run One Stage 1a Public Config

This is a longer 2B diagnostic run:

```bash
uv run progate pilot0-lora \
  --config configs/public/qwen35_2b_stage1a_k2_seed20260511.toml
```

It uses:

```text
model: Qwen/Qwen3.5-2B-Base
dataset: yahma/alpaca-cleaned
train examples: 512
diagnostic/probe examples: 128
validation examples: 512
steps: 300
probe K: 2
update mode: AdamW unchanged
```

This run is GPU-heavy compared with the smoke test. Use the 0.8B smoke first.

## Analyze a Completed Run

For a single run:

```bash
uv run progate proposal-predictability \
  --run-dir runs/<run-id> \
  --future-windows 1,5,10,20 \
  --top-fractions 0.2 \
  --permutation-n 1000
```

Bootstrap top-bottom gaps:

```bash
uv run progate bootstrap-gap \
  --run-dir runs/<run-id> \
  --future-windows 1,5,10,20 \
  --top-fraction 0.2 \
  --bootstrap-n 1000
```

For multi-seed results, collect run directories into a summary directory or use
the same internal format described by the output files. The public repository
currently ships a compact result summary rather than raw multi-seed run logs.

## Current Public Result Summary

See:

```text
docs/results/stage1a_summary.md
```

Short version:

```text
Qwen3.5-2B, Alpaca, val512, five seeds.
K=2 and K=4 both show negative top-bottom gaps at w=5,10,20.
K=2 is stronger on rank/AUC and is the current cost/signal choice.
```

This supports the diagnostic claim that AdamW proposed updates contain
measurable pre-application information about near-future validation loss
movement. It does not establish a validated optimizer improvement.

## Notes on Scope

The public repo is meant to be runnable and inspectable, not a full dump of all
private experiment planning. Omitted items include:

```text
large raw run logs
draft manuscripts
private research notes
unfinished control-policy configs
large-model experiment configs
student-report files
```

The core source code and minimal runnable configs are included so that the
diagnostic path can be inspected and executed.

## Terminology

Preferred Korean terms for reports:

```text
optimizer proposal        -> 제안 업데이트
probe data                -> 진단용 데이터
probe score               -> 진단 점수
future validation delta   -> 이후 검증 손실 변화
top-bottom gap            -> 상·하위 집단 간 차이
control policy            -> 제어 정책
extra evidence            -> 추가 배치 그래디언트 / 추가 계산 근거
```

Keep model names, dataset names, LoRA, AdamW, AUC, Spearman, and file paths in
English when needed.
