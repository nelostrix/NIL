# streaming

Train incremental classifiers/regressors by batches and resume complete checkpoints.

## Usage

```python
import nil

model = nil.streaming.create(
    nil.model("sgd_classifier", loss="log_loss", random_state=7),
    preprocess=nil.preprocessing.scaler(), classes=[0, 1])
model.partial_fit([[-2.], [-1.], [1.], [2.]], [0, 0, 1, 1])
model.checkpoint("./online-v1")
resumed = nil.streaming.resume("./online-v1", trusted=True)
resumed.partial_fit([[-3.], [3.]], [0, 1])
print(resumed.predict([[-4.], [4.]]))
```

## Parameters

`create(estimator, preprocess=None, classes=None)` requires native `partial_fit`.
Classification requires the complete label vocabulary up front. `partial_fit`
accepts a batch, targets and optional sample weights. `checkpoint(path)` creates
a new artifact directory; `resume(path, trusted=True)` restores it.

## Output

A native sklearn-protocol estimator with `batch_count_`, predictions and NIL
fit/evaluate/save support. Checkpoints include model, preprocessing and progress.

## Requirements

`nil-sdk[classical]`. Additional encoder dependencies depend on preprocessing.
No engine is imported until creation or resume is requested.

## Limitations

Preprocessing fits on the first batch and stays frozen. Supply a representative
first batch; no evolving category vocabulary or drifting feature coordinates is
claimed. A failed batch may partly mutate native state, so further updates,
predictions and checkpointing are blocked until restart or checkpoint restore.
Resume does not replay data or guarantee exactly-once delivery; the caller must
track its input offset. `fit` deliberately resets state. Checkpoints are trusted
Python artifacts, not safe exchange formats for untrusted sources.
