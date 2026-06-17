# Stage 1a Public Result Summary

This is a compact summary of the current diagnostic evidence. Raw run
directories are not included in the public repository because they are large and
still being curated.

## Setting

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
```

## K=2

```text
w=5:  Spearman -0.279, AUC 0.636, Gap@20 -0.00378, 95% CI [-0.00583, -0.00190]
w=10: Spearman -0.250, AUC 0.631, Gap@20 -0.00570, 95% CI [-0.00871, -0.00300]
w=20: Spearman -0.143, AUC 0.554, Gap@20 -0.00674, 95% CI [-0.01096, -0.00307]
```

## K=4

```text
w=5:  Spearman -0.200, AUC 0.581, Gap@20 -0.00324, 95% CI [-0.00563, -0.00121]
w=10: Spearman -0.152, AUC 0.568, Gap@20 -0.00463, 95% CI [-0.00819, -0.00164]
w=20: Spearman -0.083, AUC 0.534, Gap@20 -0.00581, 95% CI [-0.01015, -0.00176]
```

## Interpretation

Both K=2 and K=4 show negative top-bottom gaps under a 512-example validation
pool. K=2 is stronger on rank/AUC and is the current cost/signal choice.

These results support the diagnostic claim that AdamW proposed updates contain
measurable pre-application information about near-future validation loss
movement. They do not establish a validated optimizer improvement.
