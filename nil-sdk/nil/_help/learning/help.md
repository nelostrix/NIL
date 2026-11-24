# learning

## Usage

```python
import nil
# selected = nil.learning.query_uncertain(model, pool, budget=10)
# model = nil.learning.self_training(estimator, X, labels_with_minus_one)
# loss = nil.learning.contrastive_loss(first_view, second_view)
memory = nil.learning.ReplayMemory(100, seed=7)
memory.add({'feature': [1,2], 'label': 0})
```

## Parameters

Active budget/exclusions; self-training threshold/iterations and -1 unlabeled rows; contrastive temperature; support/query embeddings; replay capacity/seed.

## Output

Queried indices, fitted native estimator, differentiable NT-Xent loss, prototype labels/distances or resumable reservoir records.

## Requirements

Replay uses standard library; classical methods use explicit classical extra; contrastive loss uses Torch.

## Limitations

No automatic label oracle or guaranteed quality improvement. Two-view negatives are batch-local. Replay is a primitive, not a complete forgetting benchmark; snapshots have16MiB loading limits.

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Meta and continual learning

adapt(model.eval(), support_x, support_y, loss, lr=.01) returns functional adapted parameters with first/second-order gradients; requires buffer-free modules and at most 1M parameters. continual_metrics(stage_by_task_matrix) reports forgetting and backward transfer; use None for future tasks. It does not train a continual model.
