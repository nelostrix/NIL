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

No automatic label oracle or guaranteed quality improvement. Two-view negatives are batch-local. ReplayMemory is a primitive; the separate continual_fit workflow provides staged fitting/evaluation; snapshots have16MiB loading limits.

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Meta and continual learning

adapt(model.eval(), support_x, support_y, loss, lr=.01) returns functional adapted parameters with first/second-order gradients; requires buffer-free modules and at most 1M parameters. continual_metrics(stage_by_task_matrix) reports forgetting and backward transfer; use None for future tasks. The metric function itself does not train a model; continual_fit below provides the separate staged trainer.

## Continual classification workflow

```python
# tasks = [{'id': 'first', 'train': (train_x, train_y),
#           'test': (heldout_x, heldout_y)}, ...]
# result = nil.learning.continual_fit(model, tasks, classes=2,
#     output_dir='/artifacts/new-run', replay_capacity=256, epochs=5)
# print(result['accuracy'], result['metrics'])
```

CPU float32/64 rank2 inputs, int64 labels, fixed feature/class dimensions. The shared trainer fits each stage with optional reservoir replay; held-out labels never enter training/replay. AdamW resets at each stage. stop_after is a total stage target; resume=True validates data/config/replay/structure before loading weights. Committed-stage recovery recomputes incomplete stages. Limits:32 tasks,1M total data elements,128 features,4096 replay rows,1M parameters. No arbitrary-step recovery, automatic task inference or guaranteed retention; replay adds compute. ReplayMemory.state_dict/from_state_dict provide owned portable replay records. Full contracts/examples: docs/CONTINUED_WORKFLOWS.md.
