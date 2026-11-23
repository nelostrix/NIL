# NIL installation contract

Installing the base NIL wheel adds exactly one distribution: `nil-sdk`.
Both supported project roots explicitly declare `dependencies = []`.
NIL never runs an installer or downloads a missing backend when an API is called.
Importing NIL, listing models and reading bundled help require only Python's
standard library. They do not start services or load compute frameworks.

## Install the built wheel

Extract the distribution bundle, then run:

```sh
python -m pip install --no-index ./wheels/nil_sdk-0.2.1-py3-none-any.whl
python -m nil models
python -m nil help --list
python -m nil doctor
```

This uses normal dependency resolution. No `--no-deps` workaround is needed.
Python and pip must already exist. Wheel installation avoids source-build tools.
Installing from source can provision setuptools/wheel in a temporary build
environment; those tools are not NIL runtime dependencies.

## Current computation boundary

The base install does **not** contain native replacements for scikit-learn,
PyTorch or XGBoost. Their current model adapters cannot train without their
engines. Calling an unavailable adapter raises `MissingDependencyError` with an
explicit explanation; it does not install anything or return fabricated results.
The `nil.sklearn` API mirror likewise requires the scikit-learn backend when an
API is requested. Namespace access does not convert it into a native NIL engine.

Extras such as `nil-sdk[classical]`, `nil-sdk[torch]` and `nil-sdk[all]` are
deliberate requests to install external packages and their dependencies. Do not
select extras when you require a NIL-only environment. Dependencies in a
developer's test environment are separate from the base wheel's requirements.
For explicitly requested CPU XGBoost use `nil-sdk[boosting-cpu]` where a compatible
CPU wheel exists. This adds the CPU engine and sklearn only when selected; it
does not change the base install. The full boosting extra selects the standard
XGBoost distribution. Choose a single variant for a given environment. See
[XGBoost's installation guide](https://xgboost.readthedocs.io/en/stable/install.html).

Supporting all training with NIL alone requires implementing and validating
NIL's own numerical/model engines. Packaging engines inside NIL would retain
their storage and memory costs. The current dependency-free installation is
verified; dependency-free universal AI training is not implemented.

## Release verification

From the repository, with the development build tools already available:

```sh
python scripts/build_sdk_bundle.py --output dist
```

The build runs `scripts/verify_core_install.py` before creating the bundle.
The verifier rejects mandatory wheel requirements, external top-level packages
and bundled native libraries. It creates a temporary Python environment,
records its packages, installs the wheel offline with pip's resolver enabled,
and requires the only addition to be `nil-sdk`. It then checks installed-wheel
imports, offline help, CLI commands and missing-backend behavior. Temporary
files are removed on exit. Existing developer environments are not modified.

To recheck an extracted bundle independently:

```sh
python scripts/verify_core_install.py ./wheels/nil_sdk-0.2.1-py3-none-any.whl
```

This check measures installation and import isolation, not model speed, peak
memory or compatibility with a particular old PC.
