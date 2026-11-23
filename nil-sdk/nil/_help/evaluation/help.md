# evaluation

Evaluate classical or neural models with held-out, sample-weighted metrics.

## Usage

```python
import nil
import torch

data = nil.data.tensors(torch.zeros(4, 1), torch.zeros(4, 1))
model = nil.nn.mlp(1, 1, hidden=())
print(nil.evaluate(model, data, device_name="cpu"))
```

## Parameters

`model`, `data`, neural `task` (regression/classification/binary_classification/
multilabel_classification), batch_size and device_name. Integer classification
ignores -100 labels. BCE tasks require matching finite binary targets.
For native RL models/results, pass a separate Gymnasium environment with
episodes/max_steps/seed/deterministic options; neural batch/device arguments do
not apply. The operation dispatches without loading unrelated backends.

## Output

Neural loss/accuracy, binary/multilabel micro-F1 and multilabel subset accuracy;
classical metrics; or bounded RL raw episode-return summaries and step counts.

## Requirements

nil-sdk[classical] for classical metrics; nil-sdk[torch] for neural metrics.

## Limitations

nil.evaluate(model_or_result, X, y) also handles classical supervised models with MAE/MSE/RMSE/R2 or accuracy/precision/recall/F1/confusion/probability metrics. A tiny example does not establish model quality.
