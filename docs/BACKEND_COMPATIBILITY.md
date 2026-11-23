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
claiming the oldest patch of every package. All four hosted jobs use Linux and
Python 3.12. JSON reports record actual resolved versions, OS and interpreter;
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
not establish a minimum/current matrix for all of them.

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
