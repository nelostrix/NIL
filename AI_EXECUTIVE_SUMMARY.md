# AI Executive Summary

Audit date: 2026-11-14. Scope: NIL at `ed7f987ced82bbb59c8373c1d9bd3d9dd7cd8563` and adjacent NELO Studio IDE at `e517c6e02cd1e8bf59fccaa8c947143cae67537c`. Reports reside in NIL because that is the writable workspace. NELO Studio **is the IDE**; NIL is an agent SDK/backend, not the complete IDE. Existing audit documents were not treated as test evidence. No existing source code was modified.

Evidence is local source inspection and explicitly listed safe probes. No external API requests, dependency downloads, model downloads, cloud jobs, deployments or hardware access were performed. Optional installed runtimes and release bundles may differ from these checkouts; their functionality is not assumed.

NELO Studio **is an IDE**, with real editor, language/debug/terminal integration code and substantial AI/robotics backend work. NIL supplies an agent SDK and daemon. The combined platform is **not production-ready as a complete AI development SDK** on the inspected evidence.

Weighted completeness: **25.71/100**. Verified-only evidence floor: **4.04/100**. The matrix contains 79 scored capability rows. Weights: IDE 12%, data 10%, ML 13%, generative AI 13%, agents 13%, multimodal 6%, optimization 7%, MLOps 10%, security 10%, robotics 6%. Credit is verified 1.0, untested 0.5, partial 0.25, broken/placeholder/absent 0; equal weight per row within each area. The low verified floor reflects limited executed evidence and missing dependencies, not proof that all untested code fails.

## What is substantiated

Source tracing confirms real provider adapters, a cognitive/tool execution loop, persistent SQLite memory, native IDE integrations, graph/export workers and external STACC HTTP bridges. Safe probes verify narrow tool dispatch, SQLite persistence and model-selection validation. Three existing packaging tests pass. These successes do not verify a complete model development workflow.

## Release blockers

- Plan mode reaches the action-capable daemon without enforcing read-only policy.
- NIL dispatch lacks runtime type/effect enforcement; typed joint arrays generate incorrect schemas.
- Robotics safety claims exceed position/velocity filtering; IK failure can return an actionable zero pose.
- A separately runnable legacy daemon and compute runtime fabricate operational success/model state.
- Public NIL import fails without undeclared NumPy. The default IDE training/notebook/serving runtime nelo_train is missing; artifact and ONNX tests fail at import.
- SDK/IDE storage and action-governance paths are fragmented. Provider fallback identity and context propagation are inconsistent.

## Verification limits

Python 3.14.10 is present; Node.js, pytest, NumPy, ML frameworks and provider libraries are absent. NIL's 166 Python files yielded one syntax failure in the vision module. No frontend build, model training, live provider request, GPU measurement, deployment or hardware test was performed. Existing benchmark files and UI labels were not counted as verified functionality.

## Recommended decision

Treat the current platform as an engineering alpha. Close P0 defects, repair packaging and public API contracts, restore versioned runtime packages, then prove a tiny CPU data-to-serving lifecycle. Reuse PyTorch/TensorFlow/sklearn, established inference backends, MLflow and dataset versioning tools. Add advanced autonomy and optional optimization only after the common policy, artifact and evaluation boundaries are reliable. The 20-task roadmap includes approach, dependencies, acceptance, complexity and tests for every task.

## Reports

- [Full technical audit](AI_SDK_AUDIT.md)
- [Feature matrix and scoring](AI_FEATURE_MATRIX.md)
- [Architecture review and diagrams](AI_ARCHITECTURE_REVIEW.md)
- [Missing and incomplete capabilities](AI_MISSING_FEATURES.md)
- [Tests and limitations](AI_TEST_REPORT.md)
- [Prioritized roadmap](AI_DEVELOPMENT_ROADMAP.md)
