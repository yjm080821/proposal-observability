# Forward-Only Proposal Diagnostics

This repository studies whether an optimizer's proposed update can be inspected
before it is applied during LoRA adaptation of language models.

Student-facing reports should use neutral method names such as:

```text
forward-only proposal diagnostic
proposal diagnostic score
optimizer proposed update
제안 업데이트 진단
순전파 기반 진단
```

The current focus is experiment-first: verify whether the signed AdamW update
proposal contains information about near-future validation loss movement, then
only later test whether that signal can improve online training control.

## Current Status

The main active experiment is Stage 1a on Qwen3.5-2B LoRA adaptation with a
larger validation pool.

```text
model: Qwen/Qwen3.5-2B-Base
dataset: yahma/alpaca-cleaned
train examples: 512
diagnostic/probe examples: 128
validation examples: 512
steps: 300
seeds: 20260511, 20260512, 20260513, 20260514, 20260515
update mode: AdamW unchanged
control mode: off
K tested so far: 2, 4
K deferred: 8
```

The current result is diagnostic, not an optimizer improvement claim.

K=2, five seeds:

```text
w=5:  Spearman -0.279, AUC 0.636, Gap@20 -0.00378, 95% CI [-0.00583, -0.00190]
w=10: Spearman -0.250, AUC 0.631, Gap@20 -0.00570, 95% CI [-0.00871, -0.00300]
w=20: Spearman -0.143, AUC 0.554, Gap@20 -0.00674, 95% CI [-0.01096, -0.00307]
```

K=4, five seeds:

```text
w=5:  Spearman -0.200, AUC 0.581, Gap@20 -0.00324, 95% CI [-0.00563, -0.00121]
w=10: Spearman -0.152, AUC 0.568, Gap@20 -0.00463, 95% CI [-0.00819, -0.00164]
w=20: Spearman -0.083, AUC 0.534, Gap@20 -0.00581, 95% CI [-0.01015, -0.00176]
```

Interpretation:

```text
Both K=2 and K=4 show negative top-bottom gaps under val512.
K=2 is stronger on rank/AUC and has the better current cost/signal tradeoff.
K=8 is deferred unless later experiments specifically need a higher-K ablation.
```

## Core Idea

At training step `t`, AdamW proposes a signed LoRA parameter displacement
`q_t`. Instead of immediately trusting that proposal, the diagnostic temporarily
applies it to the trainable LoRA parameters and measures loss change on separate
diagnostic batches.

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

## Current Report Artifacts

Student-report draft:

```text
personal/딥러닝_프로젝트_보고서_초안.txt
```

Current figures:

```text
docs/paper/aggressive_reanalysis_20260615/report_figures/stage1a_k2_k4_gap_auc.svg
docs/paper/aggressive_reanalysis_20260615/report_figures/stage1a_seed_gap_lines.svg
docs/paper/aggressive_reanalysis_20260615/report_figures/stage1a_final_val_by_seed.svg
```

The report describes the method as a forward-only diagnostic for optimizer
proposed updates.

## Key Documentation

```text
docs/paper/aggressive_experiment_plan_20260615.md
docs/paper/aggressive_context_handoff_20260615.md
docs/paper/aggressive_related_work_positioning_20260615.md
docs/paper/aggressive_reanalysis_20260615/
docs/sessions/2026-06-15_0325_aggressive-experiment-reboot.md
docs/research_journal/2026-06-15.md
```

Previous PRAI manuscript work remains under `paper/`, but it is not the active
research direction unless explicitly revived.

## Setup

```bash
uv sync
```

Useful smoke command:

```bash
uv run progate smoke-model --model Qwen/Qwen3.5-2B-Base --stage proposal
```

Run one LoRA diagnostic config:

```bash
uv run progate pilot0-lora --config configs/aggressive/stage1a_qwen35_2b_alpaca_val512_k2_seed20260511.toml
```

Launch Stage 1a sequence:

```bash
bash scripts/launch_stage1a.sh
```

The launcher writes logs under:

```text
docs/paper/aggressive_reanalysis_20260615/stage1a_launch_logs/
```

## Analysis Commands

Proposal predictability:

```bash
uv run progate proposal-predictability \
  --run-dir docs/paper/aggressive_reanalysis_20260615/stage1a_k2_5seed \
  --output-dir docs/paper/aggressive_reanalysis_20260615/stage1a_k2_5seed/proposal_predictability_n2000 \
  --future-windows 1,5,10,20 \
  --top-fractions 0.2 \
  --permutation-n 2000
```

Bootstrap top-bottom gap:

```bash
uv run progate bootstrap-gap \
  --run-dir docs/paper/aggressive_reanalysis_20260615/stage1a_k2_5seed \
  --future-windows 1,5,10,20 \
  --top-fraction 0.2 \
  --bootstrap-n 2000
```

## Run Outputs

Each executable run should write:

```text
runs/<run-id>/
  config.toml
  env.json
  metrics.jsonl
  summary.json
  run_note.md
```

Do not leave an experiment only in terminal history. If a run changes the
research direction, update the session note, daily journal, and handoff document.

## Next Experiments

Near-term:

```text
1. Use Stage 1a K=2 as the current cost/signal winner unless K=8 is needed.
2. Smoke-test selective control modes on 0.8B if not already verified.
3. Regenerate Stage 2 configs with the chosen K.
4. Compare AdamW, random extra compute, fixed extra compute, longer AdamW,
   and score-triggered selective compute.
```

The main Stage 2 question is not whether the diagnostic score exists. Stage 1a
already supports that. The next question is whether this score can allocate
extra computation better than simple baselines.

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
