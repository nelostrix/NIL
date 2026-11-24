# Chunked speech, continual learning and heterogeneous graphs

This phase extends [the seven foundation workflows](SEVEN_WORKFLOWS.md) and
[the advanced phase](ADVANCED_WORKFLOWS.md). It closes three specific gaps while
keeping the 32-area framework backlog open. Installation remains NIL-only by
default. None of these modules starts a microphone, model download, background
worker, service or backend probe on import.

| Workflow | Public API | Explicit backend |
|---|---|---|
| Incremental fixed-window speech | `nil.audio.stream`, `nil.audio.load_stream` | NumPy; actual native ASR additionally needs local weights, Torch and Transformers |
| Staged continual classification | `nil.learning.continual_fit` | CPU Torch and NumPy |
| Typed relational node learning | `nil.graphs.hetero_sage`, `relation_mean`, `save_hetero`, `load_hetero` | Torch |

Use `nil-sdk[torch]` to install the explicit numerical backend. Native local speech
loading uses the existing `generative` extra. Extras install their declared
packages; installing base NIL does not install these backends. Existing workflows,
module imports and artifact formats remain supported.

## Incremental speech: push, flush and restart

```python
import nil

recognizer = nil.audio.load_asr('/models/local-ctc')
stream = nil.audio.stream(recognizer, window_seconds=5., min_tail_seconds=.1)
# blocks is a caller-owned iterable of normalized mono Waveforms at the model rate.
for block in blocks:
    for record in stream.push(block):
        print(record['start_seconds'], record['end_seconds'], record['text'])
for record in stream.flush():
    print(record['text'])
```

`push` accepts arbitrary block boundaries. It decodes every complete fixed window,
returns newly completed records, and retains less than one window of samples.
Offsets use absolute integer sample counts; floating seconds are derived from the
sample rate, avoiding cumulative timestamp drift. Records contain text, start/end
sample, start/end seconds and the recognizer fingerprint. These are **window**
intervals, not word timestamps or detected utterance boundaries.

| Parameter | Default | Contract |
|---|---|---|
| `window_seconds` | 5 | Finite0.1–30seconds; must contain at least one sample |
| `min_tail_seconds` | 0.1 | Finite0..window duration; a shorter final tail raises |
| `max_windows_per_push` | 64 | Integer1–256; split larger pushes explicitly |
| Input | — | Finite normalized float mono audio, exact recognizer sample rate |

`flush()` decodes a final partial window and closes the stream. An empty flush
closes it without emitting a fabricated record. A tail below the explicit minimum
raises and leaves the stream open, so more samples can be appended. There is no
implicit padding, dropping, channel conversion or resampling. Native backends may
require longer minimum input durations; choose the tail budget accordingly.

```python
# Save between pushes, including pending samples and the absolute offset.
artifact = stream.save('/artifacts/new-speech-state')
resumed = nil.audio.load_stream(artifact, recognizer=recognizer)
```

Snapshots are immutable bounded JSON artifacts. They contain pending audio samples,
window configuration, offset, closed state and model identity. Reload rejects a
changed model identity/rate. A closed snapshot remains closed. Large high-rate
pending windows can exceed the shared16MiB snapshot limit; use shorter windows if
portable snapshots are required. Recognizers must be pure/deterministic and expose
`transcribe(Waveform)`, `sample_rate` and a stable `fingerprint`.

A decoder failure leaves pending samples and counters unchanged. Retrying may call
the recognizer again, so side-effecting callbacks are outside this contract.
Delivering returned records and saving stream state to an external system is not
one atomic transaction: the application owns downstream acknowledgement/deduplication.
The tuple `(model_fingerprint, start_sample, end_sample)` can identify a window
within a caller-identified recording/session. It is not globally unique across
recordings.

This is incremental **fixed-window inference**, not a recurrent-state ASR decoder.
Words can be split or misrecognized at boundaries. There is no overlapping-window
text merger, VAD, diarization, microphone capture or guarantee of real-time latency.
The recorded CTC probe verifies block invariance/restart on real audio, with WER
unmeasured because no gold transcript is supplied.

## Continual learning: fit stages, evaluate forgetting, resume

```python
import torch
import nil

# Each task supplies independent train and held-out test tensors.
# X: finite CPU float32[N,features]; y: CPU int64[N], values 0..classes-1.
tasks = [
    {'id': 'first', 'train': (x1, y1), 'test': (heldout_x1, heldout_y1)},
    {'id': 'second', 'train': (x2, y2), 'test': (heldout_x2, heldout_y2)},
]
model = torch.nn.Sequential(torch.nn.Linear(3, 12), torch.nn.Tanh(),
                            torch.nn.Linear(12, 2))
partial = nil.learning.continual_fit(
    model, tasks, classes=2, output_dir='/artifacts/new-continual-run',
    epochs=10, batch_size=16, learning_rate=.03, replay_capacity=32,
    seed=7, stop_after=1,
)
result = nil.learning.continual_fit(
    model, tasks, classes=2, output_dir='/artifacts/new-continual-run',
    epochs=10, batch_size=16, learning_rate=.03, replay_capacity=32,
    seed=7, resume=True,
)
print(result['accuracy'], result['metrics'])
```

Each stage trains current examples plus the existing deterministic reservoir using
NIL's shared `Trainer`. After fitting, it evaluates all learned tasks' held-out
splits and inserts **only current training examples** into replay. There is no
label oracle and no automatic train/test split. Test labels never enter gradients
or replay. The caller remains responsible for dataset leakage and task design.

AdamW is deliberately reset at each stage, using the existing trainer's default
weight decay. Data-order/model RNG is seeded by `(seed + stage) modulo2**32`.
This is a declared rehearsal baseline, not optimizer continuity across tasks or
a universal continual-learning algorithm. `replay_capacity=0` disables replay.
More replay creates more training examples/updates per epoch; equal epochs do not
mean equal computation. Class-incremental head expansion, variable feature widths,
image/sequence inputs, arbitrary custom objectives and GPU continuation are open.

| Parameter | Default | Contract |
|---|---|---|
| `classes` | required | Integer2–256, fixed output head across stages |
| `epochs` | 5 | Integer1–100 per stage |
| `batch_size` | 32 | Integer1–1024 |
| `learning_rate` | .001 | Finite(0,1] |
| `replay_capacity` | 256 | Integer0–4096 original examples |
| `seed` | 0 | Integer0..2**32-1 |
| `stop_after` | all stages | Total number of completed stages, including existing progress |
| `resume` | false | Existing run required; config/data/model structure must match |

Budgets:1–32 uniquely named tasks;1–10,000 rows per split;1–128 features;
at most one million total input/label elements and one million model parameters.
Features and parameters use a matching uniform float32/64 dtype on CPU. The model
must return finite `[batch,classes]` logits. Input tensors are copied after validation
to keep stage data stable against later caller mutation. Concurrent model mutation
or concurrent run writers are unsupported.

Return values include `status` (`completed` or `paused`), model, completed stage
count, accuracy matrix, forgetting metrics, committed checkpoint, replay state and
versioned data/config contract. Matrix row i evaluates tasks0..i after stage i;
future tasks are `None`. Metrics on a paused run cover learned tasks only. Negative
forgetting indicates improvement, not an error.

### Recovery and publication

The run contract binds ordered task IDs, every train/test tensor's content, classes,
hyperparameters, replay capacity and seed. Resume validates these, replay records,
stage counter and metric history **before** changing model weights or RNG.
Checkpoint loading uses the existing restricted state-dict/integrity/structure
checks and exact supported Python/Torch environment contract.

Each completed stage gets a unique checkpoint. An atomic `state.json` update commits
its pointer last. A crash before that pointer commits leaves the previous stage
available; resume restores that stage and recomputes the incomplete stage. An orphan
future-stage checkpoint cannot overwrite the committed stage. Invalid committed
checkpoint integrity still raises rather than silently selecting a different result.
Failure during initialization before the first state commit requires a new output
directory. Run directories/checkpoints from interrupted attempts remain as evidence;
there is no automatic garbage collector or cross-process writer lock.

Recovery is **stage-boundary**, not arbitrary batch or optimizer-step continuation.
Trainer epoch checkpoints remain available, but this wrapper intentionally resumes
from the last committed full stage. Exact continuation is tested on deterministic
CPU fixtures, not promised across hardware, backend versions or nondeterministic
custom model behavior. A failed call may leave the passed model partly trained;
call resume to restore the committed stage.

## Heterogeneous graphs: types, relations, learning and reload

```python
import torch
import nil

relation = ('author', 'writes', 'paper')
features = {
    'author': torch.tensor([[1., 0.], [0., 1.]]),
    'paper': torch.zeros(3, 1),
}
edges = {relation: torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.int64)}
model = nil.graphs.hetero_sage(
    input_dims={'author': 2, 'paper': 1}, hidden_dim=8,
    output_dims={'author': 2, 'paper': 2}, relations=[relation],
)
logits_by_type = model(features, edges)
artifact = nil.graphs.save_hetero(features, edges, '/artifacts/new-typed-graph')
restored_graph = nil.graphs.load_hetero(artifact)
nil.training.checkpoint.save('/artifacts/typed-model.pt', model)
# Recreate the same model schema before checkpoint.load(...).
```

The network uses two layers of per-type self transforms plus separate transforms
for incoming relations. The first layer has ReLU. For each `(source, relation,
target)`, incoming source features are averaged separately for each target node;
a bias-free relation transform maps that mean to the target width. Relation outputs
are summed with the target self transform. Zero-degree relations contribute zero.
Repeated COO edges count repeatedly; reverse relations must be supplied explicitly.
No edge weights or attention are inferred. This is a small relation-mean,
GraphSAGE-style variant, not a reproduction of every method in the
[GraphSAGE paper](https://arxiv.org/abs/1706.02216).

The linear relation transform commutes with the mean. NIL chooses whether to
project before or after aggregation based on the smaller intermediate, avoiding
unnecessary wide projections over a large source type. There is no dense adjacency.

Features are a type-to-tensor dictionary. Edges are keyed by unique three-string
relation tuples, each with an int64 `[2,edges]` tensor. Source/target endpoints use
local indices within their respective node types. Node types can have different
feature widths but must share device/dtype. Empty declared relations use shape
`[2,0]`; missing/extra model relations raise. Type names and relation identities
are preserved in checkpoint structure metadata, so renaming a relation is rejected
before loading compatible-shaped but semantically different weights.

Budgets:16 node types,64 relations,100,000 total nodes,one million feature/edge/
activation elements and one million model parameters; hidden width at most512.
Dimensions may be1–4096 within the total budgets. Intermediates that exceed the
explicit projection/message budget raise. Float32/64 graph JSON preserves dtype;
GPU graph/model execution is not hardware-validated in this phase. This implementation
does not add temporal graphs, distributed loaders, typed mini-batching, heterogeneous
neighbor sampling, attention layers or scalable link-prediction training.

## Runnable validation and interpretation

```sh
python nil-sdk/examples/continued_workflows.py --output /external/new-example
# Explicit local model + WAV, never a repository ID or automatic download:
python nil-sdk/examples/continued_workflows.py --output /external/new-real-example \
  --asr-model /models/local-ctc --audio /data/recording.wav
python -m pytest nil-sdk/tests/test_speech_stream.py \
  nil-sdk/tests/test_continual_workflow.py nil-sdk/tests/test_heterogeneous_graphs.py -q
```

The example reports three seeds for replay/no-replay, raw accuracy matrices,
forgetting, optimizer updates and example exposures. It also trains a synthetic
typed graph using only four of eight paper labels, validates held-out labels and
reload, and compares recognition from one push versus interrupted arbitrary-block
pushes. Typed graph topology is visible during fitting: this is transductive,
not evidence of performance on unseen real-world graphs. Synthetic held-out tasks
do not establish broad continual-learning quality.

The raw report records remaining forgetting and reduced current-task accuracy
where replay trades off retention. Baselines use equal epochs but **80 versus100**
updates and1,280 versus1,600 example exposures, so this is not a compute-matched
research comparison. Early five-epoch fixtures failed the requested current-task
quality floor; those outcomes remain in evidence. The final contextual fixture
uses ten epochs rather than claiming the low-budget attempt succeeded.

Full scope, numerical results, package sizes, environment versions and any hosted
verification are recorded in [the phase evidence](benchmarks/2026-11-24-continued-workflows.json).
The [master TODO](project/TODO.md) retains the broader unfinished AI-framework work.

## Measured synthetic baseline results

| Seed | No-replay final accuracy | Replay32 final accuracy | No-replay forgetting | Replay32 forgetting |
|---|---|---|---|---|
| 7 | 70.31% | 78.12% | 46.88 points | 21.88 points |
| 11 | 77.34% | 86.72% | 42.19 points | 20.31 points |
| 19 | 71.09% | 80.47% | 42.19 points | 29.69 points |

These two-stage tasks have64 training and64 held-out examples per stage, an explicit task-context feature, and a12-unit tanh hidden layer. Across these seeds, final average accuracy is72.92% without replay and81.77% with replay; mean forgetting is43.75 versus23.96points. Replay still forgets and can reduce current-task accuracy. These results use the unequal update/exposure budgets above, and do not establish results on real continual-learning datasets.

## Local acceptance

Full CPU SDK regression: **1,208 passed /32 optional skips**,117.38s. Installed regular-wheel acceptance outside the checkout: **212 Torch tests plus10 examples,99 current-classical tests plus5 examples,97 minimum-classical tests plus5 examples**, all with zero skips. Base wheel: **601,401 bytes**,225 Python files,114 offline help guides, zero mandatory dependencies; the offline NIL-only dependency-resolution/import audit passed. Hosted verification is recorded separately in the evidence file and is not inferred from local passes.

The synthetic seeds and quality floors were used during development. They are functional/development holdouts, not a pristine external research benchmark. ASR WER, temporal/large-scale heterogeneous learning, arbitrary-step continual recovery and actual GPU/older-PC acceptance remain open.
