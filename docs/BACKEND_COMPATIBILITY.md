# Installed backend compatibility

The dependency-free base wheel has passed Linux, macOS and Windows checks on
Python 3.10–3.12. Optional-backend acceptance is scoped separately; a successful
base job does not certify Torch, scikit-learn or every optional extra on that OS.

`scripts/accept_backends.py` verifies regular wheel installations from a temporary
working directory using isolated Python, with PYTHONPATH removed. Only tests,
fixtures and examples are copied; SDK implementation comes from the installed
wheel. A source/editable installation, missing dependency, empty test set, skipped
test, failed example or per-step timeout fails the profile and preserves logs.
CUDA is explicitly excluded from CPU profiles, not counted as accepted.

| Profile | Explicit minimum-family configuration | Current configuration | Contract |
| --- | --- | --- | --- |
| Classical | scikit-learn 1.4.2, NumPy 1.26.4, SciPy 1.13.1, pandas 2.3.3 | Resolved classical/tabular extras | Pipelines, held-out metrics, CV/search, inspection, API mirror, two installed examples |
| Torch CPU | Torch 2.6.0 CPU, NumPy 1.26.4 | Resolved CPU Torch >=2.6,<3 and NumPy >=1.26,<3 | Training/gradients, accumulation, supported resume, isolated inference, TLS service/credentials, CNN/LSTM/Transformer examples |

These are exact declared profiles, not every permutation allowed by dependency
ranges. The classical profile uses the maintained 1.4 patch family rather than
claiming the oldest patch of every package. The expanded profiles use Python3.12: classical runs on Linux/macOS/Windows,
while CPU Torch and RL run on Linux. JSON reports record actual resolved versions, OS and interpreter;
CI preserves the wheel, report and step logs as named artifacts.

```sh
# Build once, then install into each separately provisioned backend environment.
python -m build --wheel --no-isolation --outdir dist
python -m pip install --force-reinstall --no-deps dist/nil_sdk-0.2.1-py3-none-any.whl
python scripts/accept_backends.py --profile classical --output dist/classical.json
# Or, in the environment with the explicit Torch CPU backend:
python scripts/accept_backends.py --profile torch --output dist/torch.json
```

The [backend workflow](../.github/workflows/sdk-backends.yml) defines reproducible
minimum/current installations. Current means what the configured indexes resolve
at run time; consult the report for exact versions. No backend is downloaded by
the verifier or installed by importing NIL. OpenSSL is explicit TLS test tooling.

Generative, ONNX, RL, specialized backends, GPU, other optional-backend OS/Python
combinations and public Internet deployment remain separate contracts. The
existing SDK development workflow exercises additional integrations, but does
not establish a minimum/current matrix for all of them. See the later expanded profile evidence below.

## Recorded hosted acceptance — 2026-11-23

[All four profiles passed](https://github.com/nelostrix/NIL/actions/runs/45116372373)
at 7fc1943. Classical: 67 tests with the minimum-family profile, 69 with current;
both installed examples passed in each. Torch: 70 CPU/native/service tests and all three
architecture examples passed with both 2.6.0+cpu and 2.15.0+cpu. All profiles had
zero skipped tests. [Raw profile reports](benchmarks/2026-11-23-backend-matrix.json)
include exact versions. The Torch profiles include independent multiclass/Adam gradient, optimizer,
learning and exact-continuation acceptance.

[Nine base-platform cells also passed](https://github.com/nelostrix/NIL/actions/runs/45115748776).
The [artifact index](benchmarks/2026-11-23-platform-artifacts.json) records each
built wheel/bundle artifact ID, digest, size and expiry alongside its passing job.
Download using `gh run download 45115748776 --repo nelostrix/NIL` into a scratch
directory. These are verification artifacts subject to CI retention, not a stable
release or package-index upload. The first failing run remains visible for audit.

## Expanded phase8–9 profiles

[Run45143679440](https://github.com/nelostrix/NIL/actions/runs/45143679440)
passed ten installed-wheel cells at aa3bfb8, all with zero skips:

| Profile | OS/Python | Versions | Accepted work |
| --- | --- | --- | --- |
| Classical | Linux/macOS/Windows, Python3.12 | Minimum/current | 67/69 tests and two installed examples per cell |
| Torch CPU | Linux, Python3.12 | Torch2.6.0+cpu/current | 118 tests, actual local C compilation, ND/sparse parity, service contracts and three examples |
| RL CPU | Linux, Python3.12 | SB3/contrib2.4.0, Gymnasium1.0.0, Torch2.6 / current | 36 tests, masked/recurrent/centralized/CQL contracts, exact DQN/PPO/A2C simulator continuation and PPO/SAC examples |

[Raw reports](benchmarks/2026-11-23-phase8-9-backends.json) record actual package
versions and artifact log names. The minimum RL job uses NumPy1.26.4 and the
current job resolves the declared RL-contrib extras. No core dependency changed.
The compiled CPU profile requires explicit Linux compiler/Python headers; the
base wheel/import does not. Non-Linux optional Torch/RL, other Python/backend
combinations and real GPU acceptance remain separate profiles rather than inferred
passes. [Base run45143679690](https://github.com/nelostrix/NIL/actions/runs/45143679690)
passes all nine OS/Python3.10–3.12 cells, and
[full SDK run45143679678](https://github.com/nelostrix/NIL/actions/runs/45143679678)
also passes. Base success still does not imply every optional backend combination.

## Framework/image follow-on — 2026-11-24

The explicit Torch profile now includes Pillow, framework result contracts and
image classification/checkpoint inference; minimum-family codec is Pillow10.4.0,
current codec range >=10.4,<14. No codec is added to the base wheel. Local regular-
wheel acceptance: Torch153/RL36/classical-current70/classical-minimum68 tests,
zero skips, all examples passing. See [recorded framework evidence](benchmarks/2026-11-24-framework.json).
Hosted full SDK run [45365343601](https://github.com/nelostrix/NIL/actions/runs/45365343601)
passed 1,253 tests with 46 optional-target skips. All nine Linux/macOS/Windows base jobs and all ten declared backend jobs passed
at verification commit `318e453`. The [base run](https://github.com/nelostrix/NIL/actions/runs/45365343708)
and [backend run](https://github.com/nelostrix/NIL/actions/runs/45365343865)
cover these new contracts; hardware targets remain separate.


## Seven-workflow acceptance — 2026-11-24

The installed classical profile now includes retrieval/tools/probabilistic tests
and the real-Iris/causal example. The Torch profile adds graph, contrastive, audio,
GIF/timing/fusion and actual MP4 tests plus the corresponding runnable examples.
FFmpeg/ffprobe are explicit external test dependencies installed only by those CI
jobs; the NIL wheel still has no mandatory dependencies or codec binaries.

Local profiles passed 189 Torch, 88 current-classical and 86 minimum-classical tests,
all with zero skips. The final hosted full SDK regression passed 1,294 tests with 46
optional-target skips. See [dated reports](benchmarks/2026-11-24-seven-workflows.json).
On macOS, the dead-supervisor fault case verifies existing safe refusal of active-
job identity recovery; it does not assert that unsupported recovery works. Normal
owned calls and completed-prefix plan resume have separate positive acceptance.

Final verification at `c8dfba5`: all nine base-platform and ten installed-backend
jobs passed. Hosted SDK: 1294 passed, 46 optional-target skips.
Exact versions, failures, safe-refusal limits and raw-report references are in the
[dated evidence](benchmarks/2026-11-24-seven-workflows.json).

Advanced seven-area additions and still-open requirements: [advanced workflow phase](ADVANCED_WORKFLOWS.md). New CPU checks are separate from the earlier v1 hosted matrix.

Final advanced-phase commit `9f78f3e`: all nine base-platform jobs and ten installed-backend jobs passed; hosted SDK **1321 passed /46 optional skips**. Final local regression **1198 passed /32 optional skips**. [Exact profiles and failures retained](benchmarks/2026-11-24-advanced-workflows.json).

New bounded workflows: [incremental speech, staged continual learning and heterogeneous graphs](CONTINUED_WORKFLOWS.md). Base NIL remains dependency-free; accepted scope and unfinished work are documented separately.
