# NIL CPU implementation and acceptance — 2026-11-22

This batch closes scoped software gaps after classical/specialized acceptance.
Base installation still adds only nil-sdk; optional adapters execute explicitly
installed third-party engines. `nil.native` alone is an independent numerical
reference implementation. No broad PyTorch/scikit-learn replacement claim is made.

## Supported implementation

| Area | Implemented and exercised | Boundary |
| --- | --- | --- |
| Native numerical subset | Dense 2D CPU float64 Tensor; reverse-mode autodiff; row/column/scalar broadcasting; matmul, arithmetic, powers, activation/reduction operations; stable BCE/MSE; Linear/Sequential/Tanh/ReLU; SGD momentum/decay; JSON model/optimizer persistence | 100k elements/Tensor and total network parameters; 16 layers; no sparse/GPU/other dtypes; pure Python reference performance |
| Distributed training | Two-rank CPU Gloo, global mean-loss parity with uneven rank batches, SGD momentum, rank-zero observers, exact epoch checkpoint continuation, previous epoch recovery after rank failure | Single host, finite tensor data, FP32, no mutable buffers/BatchNorm/dropout; GPU path unaccepted |
| Activation checkpointing | Nonreentrant Sequential recomputation; actual saved-activation byte reduction; gradient and model reload parity | CPU acceptance only; not FSDP |
| Offline neural RL | Discrete/continuous behavior cloning; discrete fitted-Q baseline; masks and terminal/truncation semantics; held-out learning; optimizer/target/sampler continuation | Flat finite inputs, CPU, bounded network/data; no CQL/IQL quality claim |
| Structured RL | Actual RecurrentPPO/MaskablePPO integration; recurrent evaluation reset; vector autoreset terminal observation; parallel dict multi-agent collection and independent neural learners | sb3-contrib explicit extra; masked multi-agent combination rejected; no centralized critic or arbitrary env adapter |
| Exact simulator continuation | CPU DQN plus ordinary replay, one DummyVecEnv and standard file-free Monitor; explicit raw simulator state protocol; environment/space/global RNG and vector/model counters; safe cooperative cancellation | Matching versions, identical learn-call schedule; unsupported wrappers/normalization/algorithms explicitly rejected |
| Serving | Authenticated loopback HTTP, frozen trusted CPU model in restartable process; bounded admission and zero inference queue; hard inference deadline; crash recovery; metrics; sanitized errors | Local deployment building block; external TLS/auth/proxy/orchestration and aggregate RAM/GPU quota acceptance remain open |

The numerical subset rejects nonfinite/ragged/sparse/external tensor inputs and
unsupported dtypes/devices. Leaf gradient accumulation is explicit; zero gradients
before a new update. Version checks prevent stale graphs after parameter mutation.
Finite differences exercise matrix/broadcast and nonlinear/composite derivatives;
Torch float64 parity is an independent optional check. No fallback silently installs
or imports a backend. Every public module has bundled help.md.

## Local generative/export compatibility matrix

Actual Linux/Python 3.12.16, Torch 2.15.0+cpu, NumPy 2.5.4,
Transformers 4.57.6, PEFT 0.22.0, ONNX 1.23.2, ONNX Runtime 1.32.0:

| Workflow | CPU result | Remaining limit |
| --- | --- | --- |
| Local BPE + tiny GPT2 creation, causal training, held-out perplexity, local save/reload | Passed without model downloads | Not large-model performance or pretrained-model compatibility certification |
| Local GPT2 LoRA train, merge, safetensors reload and generation | Passed | No GPU, QLoRA or every adapter architecture claim |
| Dynamic INT8 Linear quantization and output tolerance | Passed | Supported native CPU layers only; Torch emits upstream deprecation warnings |
| MLP ONNX + actual CPU runtime parity | Passed | No universal operator/export guarantee |
| CNN ONNX with different batch size | Passed | Concrete tested architecture only |
| LSTM ONNX with explicit hidden/cell and different batch/time dimensions | Passed | Other recurrent/export combinations untested; exporter emits tracing warnings |
| Activation recomputation saved tensors and gradients | Passed | Not a process peak-RSS measurement or FSDP acceptance |
| CUDA fp16/bf16, NCCL DDP, GPU FSDP distributed checkpoint | Not run | Actual GPU hardware required |

The acceptance verifier requires requested backends before running:

```sh
PYTHONPATH=nil-sdk OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/verify_neural.py --rl --contrib --generative
```

`--cuda` refuses acceptance on a CPU-only host. Native policy versions tested:
SB3/sb3-contrib 2.9.2, Gymnasium 1.4.0. Optional integrations are not NIL-native
engines. No external training data, model download or public endpoint is needed.

## Measured benchmarks

Reproduce with `PYTHONPATH=nil-sdk python scripts/benchmark_native.py --output
report.json` and `PYTHONPATH=nil-sdk python scripts/benchmark_rl.py --output
report.json`. Raw independent samples, seeds, versions and host details ship in
`docs/benchmarks/2026-11-22-native-core.json` and
`docs/benchmarks/2026-11-22-rl-quality.json`.

The native reference fits 65 one-feature rows for 150 SGD updates: median fit
**0.0958 s**, median fresh-process peak RSS **15.43 MiB** over five child processes
running Python `-S`; no NumPy/Torch/sklearn import. This is a small reference
workload, not a comparison against vectorized engines or a ten-year-old PC claim.
The documented target is final MSE below 1e-6 with no numerical dependencies.

PPO trains for 512 steps per seed on a one-step two-action deterministic reward
preference, then evaluates 100 episodes per seed. Five seeds score **1.000** vs
**0.532** for an actual uniformly random policy. Bootstrap 95% CI over seed means:
policy [1.000, 1.000], baseline [0.508, 0.556], paired improvement [0.444, 0.492].
10,000 deterministic bootstrap replicates; seeds, not correlated episodes, are the
resampling units. Five seeds and a trivial task do not establish broad RL quality,
exploration ability, sample efficiency or performance on continuous/MARL tasks.

## Local service deployment and failure contracts

1. Use an explicit, importable trusted CPU model. Start from a guarded Python
   entry point. Keep the generated Bearer token private; send no browser origin.
2. Construct `nil.serving.create_isolated_server(model, token=...)`; bind its
   returned loopback address. Call `serve_forever()` only when ready.
3. Send authenticated JSON to `/predict`; authenticated `/health` and `/metrics`
   report admission state, requests/rejections, latency, deadline/crash counts and
   worker restarts. Startup is lazy and has its own bounded deadline.
4. Respect 503 backpressure with a bounded client retry policy. 504 kills the
   overdue worker; 502 indicates a crashed worker. Next request starts fresh from
   the frozen model, so forward mutations are not durable service state.
5. Supervise the service process and configure separate gateway TLS, identity,
   total request budgets and OS memory/CPU quotas before external deployment.
   `shutdown()` then `server_close()` stops serving and its worker.

Acceptance uses actual HTTP sockets/processes, unauthorized/browser/Host rejection,
nonfinite/schema failures, concurrent admission while inference is busy, forced
model error, real over-deadline sleep, abrupt process exit and subsequent healthy
inference. No request data/model exception is logged. Frozen model deserialization
is trusted-owner code only; requests cannot supply model objects. The process is
not a security sandbox, and input/output counts are not aggregate RSS limits.
Public multiuser production deployment remains an acceptance gate.

## Remaining acceptance targets

Linux Ryzen 3 7320U has no CUDA device and is not a ten-year-old PC. Windows/macOS,
aggregate process/GPU quotas, old-PC scale benchmarks, real CUDA overflow/resume,
GPU DDP, GPU FSDP save/load and the complete OS/Python/backend matrix remain open.
Existing CI/release checks and this CPU matrix do not substitute for running those
targets. See TODO.md for checked acceptance gates and current full-suite evidence.


## Validation record

Full source suite runs in the separate environments passed 1,034 classical tests
(15 optional skips) and 967 neural/RL tests (13 optional/hardware skips). After
final output/allocation budget checks, the focused native/offline suite passes
14 tests. Installed checks outside the source tree pass 26 tests (one deliberate
source-bootstrap deselection); the installed `completion_workflows.py --neural
--ddp` example passes. A fresh offline resolver installs only nil-sdk and executes
actual native training with no compute backends. Source/wheel parity: 194 Python
files; 103 bundled help guides; approximately 476 KiB wheel. These are separate
backend environments, not a combined universal coverage score.


Final installed numerical acceptance after the normalized-storage budget fix:
**13 passed, one source-bootstrap deselection**. The installed --neural example
passes again. No additional dependency was installed during either wheel update.


Subsequent CPU FSDP implementation and target acceptance infrastructure are
recorded in [the sharding batch](benchmarks/2026-11-22-sharding-gates.md). Real
CPU partitioning/save/load/fault recovery now pass; GPU and large-model memory
acceptance remain open.
