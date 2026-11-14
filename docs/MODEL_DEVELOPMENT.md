# NIL model development SDK 0.2.1

NIL now includes a real model-development API alongside its existing agent SDK.
The compute foundation is native PyTorch: use NIL's helpers for common cases,
and ordinary tensors, modules, optimizers, data loaders and custom training steps
for research architectures. NELO Studio remains the IDE; STACC remains the
robotics execution backend. This SDK release does not replace those applications.

## Install

Use Python 3.10 or newer in a virtual environment; the acceptance environment
uses Python 3.12 and CPU PyTorch 2.6. Install a suitable PyTorch wheel for your
accelerator separately when appropriate, then install the desired NIL extras:

```sh
python -m pip install '.[torch,robotics]'
python -m pip install '.[generative,classical,onnx,dev]'
nil doctor
nil demo --output ./nil_runs
```

From an extracted distribution bundle, replace `.` with the wheel path:

```sh
python -m pip install './wheels/nil_sdk-0.2.1-py3-none-any.whl[torch,generative,onnx]'
```

The bundle includes the wheel, examples, this guide and a SHA-256 manifest. It
does not vendor Python, frameworks, GPU drivers or pretrained model weights.
Installing extras downloads libraries according to your package-manager settings.
`import nil`, local records, run storage and artifact registration work without
PyTorch. Model APIs explain which extra is missing when invoked.

## A first model

```python
import nil
import torch

nil.runtime.seed(7)  # Set before constructing a model for reproducible initialization.
x = torch.linspace(-1, 1, 96).reshape(-1, 1)
dataset = nil.data.tensors(x, 2 * x + 1)
model = nil.nn.mlp(1, 1, hidden=(32, 16))

result = nil.train(
    model, dataset, task="regression", epochs=30,
    learning_rate=0.01, device="cpu", output_dir="./nil_runs",
)
print(nil.evaluate(result.model, dataset, device_name="cpu"))
print(nil.predict(result.model, x[:2], device_name="cpu"))
```

Tasks: `regression` and `imitation_learning` use MSE; `classification` uses
cross-entropy with integer class targets; `causal_lm` consumes a model-returned
loss and explicit labels. Regression outputs and targets must have matching
shapes; implicit loss broadcasting is rejected. `nil.nn.Module`, `nil.nn.Linear`
and other torch.nn classes are the actual PyTorch classes, not reimplementations.
`nil.tensor`, `nil.optim` and `nil.autograd` also expose native PyTorch objects.

## Custom models and training objectives

Any torch.nn.Module works, including convolutional, recurrent, graph, transformer,
diffusion, time-series and multimodal architectures, provided you supply a
compatible batch and loss. This is extensibility, not a claim that NIL includes
prebuilt recipes or evaluation suites for all those model families.

```python
def step(model, batch):
    inputs, targets = batch
    return torch.nn.functional.smooth_l1_loss(model(inputs), targets)

trainer = nil.Trainer(
    model,
    task=step,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    config=nil.TrainingConfig(
        epochs=10, device="cpu", accumulation_steps=2,
        clip_grad_norm=1.0, output_dir="./nil_runs",
    ),
)
result = trainer.fit(dataset, validation=dataset)
```

Custom steps return one differentiable scalar loss tensor. The trainer averages
microbatch gradients within each accumulation window, including a final partial
window. Epoch loss is a mean of batch losses; use a custom evaluation routine
when exact token/sample weighting is required. Evaluation's regression and
classification metrics use element/sample weighting respectively.

Callbacks receive `(trainer, run, epoch_metrics)`. Call `trainer.cancel()` to
cooperatively stop between batches or at an epoch boundary. This does not kill
arbitrary stuck native code and is not a process sandbox. Nonfinite loss or
gradients stop the run; errors and completed/cancelled state are persisted.

## Checkpoint and resume

```python
result = trainer.fit(dataset)
trainer.config.epochs = 20  # Total target epoch count, not 20 additional epochs.
resumed = trainer.fit(dataset, resume=result.checkpoint)
```

Checkpoints hold model/optimizer/scheduler state, epoch, update count, history,
Torch RNG and Python RNG state. Built-in map-style loaders use seed+epoch, and
resume requires matching batch size, shuffle, accumulation and seed. A test
verifies exact CPU parameter equality after dropout/shuffled resume.

Resume is at a completed epoch boundary. Custom data loaders, NumPy RNG,
external augmentation state, model construction and custom callbacks remain
the developer's responsibility. GPU determinism is not guaranteed. The trainer
currently implements single-process training, not DDP, AMP, FSDP or DeepSpeed.

Checkpoints use state dictionaries and explicit `torch.load(weights_only=True)`.
Their adjacent `.pt.json` manifest is checked before deserialization. A hash
detects corruption; it is not an authenticity signature. Only load trusted
artifacts and preserve both files. PyTorch's [serialization documentation](https://docs.pytorch.org/docs/stable/notes/serialization)
explains the state-dictionary and restricted-loader approach.

## Language models and LoRA

```python
texts = ["a robot observes the room", "a robot stops safely"] * 16
tokenizer = nil.generative.tokenizer_from_texts(texts, vocab_size=300)
data = nil.generative.text_dataset(texts, tokenizer, max_length=24)
model = nil.generative.causal_lm(
    len(tokenizer), context_length=48, width=32, layers=1, heads=2,
)
# Optional parameter-efficient fine-tuning on this locally constructed model:
model = nil.generative.lora(model, rank=4, target_modules=["c_attn"])
result = nil.train(model, data, task="causal_lm", epochs=3, device="cpu")
```

These tiny models validate the development workflow; their generated text is not
expected to be useful. A local pretrained model directory can instead be loaded
with `nil.generative.load_causal_lm(path)`. Network downloads are disabled by
default; `allow_download=True` explicitly opts in. Remote model code is refused.
Returned models and tokenizers remain native Transformers/PEFT objects.

`nil.generative.save(model, tokenizer, path)` writes native safetensors artifacts;
a PEFT model saves an adapter requiring its original base model. To save a full
standalone model from an ordinary LoRA run, explicitly call `merge_and_unload()`
first when supported. Generation restores the model's original training mode.
Padding labels are -100 based on attention masks, so EOS tokens are not confused
with padding. This follows the [causal LM training convention](https://huggingface.co/docs/transformers/tasks/language_modeling).

For instruction tuning, supply your own chat template and response-only labels;
the convenience text dataset performs full-text causal training. RLHF/DPO,
QLoRA, GPTQ/AWQ, INT4, distributed LLM training and specialized serving are not
implemented by this release. Native libraries remain usable directly.

## Quantization, ONNX and profiling

```python
quantized = nil.optimization.quantize(result.model)  # Supported CPU layers only.
timing = nil.optimization.benchmark(quantized, example_inputs, iterations=50)
path = nil.optimization.export_onnx(
    model, example_inputs, "./model.onnx",
    input_names=["features"], output_names=["prediction"],
    dynamic_axes={"features": {0: "batch"}, "prediction": {0: "batch"}},
)
runtime = nil.inference.OnnxModel(path, providers=["CPUExecutionProvider"])
outputs = runtime.predict({"features": numpy_inputs})
```

Use a compatible native tensor-returning model here, such as the regression
model in `examples/model_development.py`. The dynamic INT8 helper supports
PyTorch Linear/LSTM/GRU on CPU and returns a separate model. It is not a general
LLM quantizer: GPT-style Conv1D layers, accelerator quantization and newer
torchao paths require architecture-specific integrations. PyTorch documents the
[dynamic quantization operation](https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_dynamic.html).
The ONNX export helper uses a cloned CPU model and validates the graph; models
with unsupported operators or custom output structures need native export code.
ONNX input dtype is explicit and is not silently coerced to float32.

## Run storage and model registry

Every run writes `run.json`, `events.jsonl` and `checkpoints/latest.pt` with its
manifest. JSON events record real metrics, status and checkpoint paths. These
files are ready for an IDE adapter; a new NELO Studio UI integration was not
implemented in this change.

```python
registry = nil.Registry("./nil_registry")
version = registry.register("policy", result.checkpoint, {"run": result.run.id})
restored = registry.load("policy", model_factory(), version=version["version"])
# Optional explicit export to an installed MLflow backend:
# nil.lifecycle.export_mlflow(result, tracking_uri="sqlite:///mlflow.db")
```

The registry stores copied files and allocates versions transactionally in SQLite.
Resolve/load verify SHA-256. Architecture construction is explicit; no arbitrary
pickled model code is loaded. Model directories, tokenizer bundles, remote object
stores, automatic promotion, rollback routing and signature validation require
further integrations. MLflow export is optional and has not been live-tested here.

## Local inference service

```python
import secrets
token = secrets.token_urlsafe(32)
server = nil.serving.create_server(model, port=8768, token=token)
# Give the token only to your trusted local client; do not log it.
server.serve_forever()  # Explicit action; close/shutdown when done.
```

The development server binds 127.0.0.1, requires a Bearer token for health and
prediction, refuses browser origins and bounds request bytes. POST `/predict`
with `{"inputs": [[1.0]]}` returns `{"outputs": ...}` for a compatible tensor
model. It is single-process, CPU-only and trusted-code-only; it does not provide
production TLS, multiuser auth, batching, isolation or bounded native model time.

## Robotics and classical models

`nil.robotics.policy(observation_dim, action_dim)` constructs an ordinary MLP;
`nil.robotics.demonstrations(observations, actions)` validates finite offline
training data. Train with `task="imitation_learning"`. No hardware command is
sent. Sensor semantics, action units, robot-specific bounds and deployment
approval remain explicit. The position/velocity filter is not a torque or
collision controller; STACC/independent controls must enforce physical safety.

For a native sklearn estimator, call
`nil.classical.fit(estimator, X, y, validation=(X_test, y_test), task="regression")`.
The returned model is the original estimator, with NIL run/metric records.
TensorFlow/Keras-specific training adapters are not included in this release.

## Verification and distribution

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest nil-sdk/tests -q
python nil-sdk/examples/model_development.py --output ./nil_runs --onnx
python nil-sdk/examples/local_language_model.py --output ./nil_runs --lora
python scripts/build_sdk_bundle.py --output dist
```

See `SDK_IMPLEMENTATION_REPORT.md` for executed checks, tested dependency versions,
scope and remaining work. The earlier AI audit documents describe the pre-change
snapshot and intentionally remain historical evidence.

## Quick additions in 0.2.1

Early stopping is an optional callback with an explicit metric, direction and
minimum improvement. It completes the run successfully at an epoch boundary:

```python
stop = nil.training.EarlyStopping(patience=3, monitor="val_loss", min_delta=0.001)
trainer = nil.Trainer(model, config=nil.TrainingConfig(epochs=100), callbacks=[stop])
result = trainer.fit(training_data, validation=validation_data)
# Model is still at the final epoch; explicitly restore the selected snapshot:
nil.training.checkpoint.load(stop.best_checkpoint, model)
```

`best.pt` contains model weights; use `latest.pt` to resume full optimizer state.
The callback resets when a new/resumed run begins. Early-stopping state does not
carry across resumed runs. Cancellation remains distinct from early completion.

For a small, bounded search over training hyperparameters:

```python
search = nil.training.grid_search(
    lambda: nil.nn.mlp(4, 1, hidden=(16,)),
    training_data,
    validation_data,
    {"learning_rate": [0.001, 0.01], "batch_size": [16, 32]},
    config=nil.TrainingConfig(epochs=5, device="cpu", output_dir="./nil_runs"),
    max_trials=4,
)
model = search.load_best(nil.nn.mlp(4, 1, hidden=(16,)))
print(search.best_config)
```

Search is sequential, trains fresh models with identical initialization seeds,
selects the final epoch's `val_loss` by default, and persists trial configurations,
errors and checkpoint paths in `search.json`. The entire grid is validated against
`max_trials` before models are created. It does not tune architecture fields,
parallelize work or guarantee deterministic behavior for custom external loaders.
Use a separate held-out test set after selecting on validation data.

LLM evaluation computes next-token cross-entropy summed over unmasked targets,
then divides by the actual target-token count before exponentiation:

```python
print(nil.generative.perplexity(model, tokenized_validation, device_name="cpu"))
# {"nll": ..., "perplexity": ..., "tokens": ...}
```

It handles right/left padding, ignores -100 labels and restores training mode.
Results describe the supplied independent windows; this is not sliding-window
full-corpus evaluation. Compare only consistent tokenizers and context/masking
policies. See the [Transformers perplexity discussion](https://huggingface.co/docs/transformers/perplexity)
for why tokenization and context affect comparisons.

CPU profiling measures a cloned model, preserving the original model:

```python
report = nil.optimization.profile(model, example_inputs, iterations=20, warmup=3)
print(report["serialized_state_bytes"], report["latency"])
```

Reports include parameter counts, parameter/buffer bytes and serialized state
size. Packed quantized weights may not appear in `parameters()`, so compare
serialized-state bytes for FP32/INT8 models. Profiling clones and serializes the
model in memory; allow adequate RAM. Latency includes NIL inference-call overhead.
Peak RAM, FLOPs, GPU performance and training throughput are not measured.
