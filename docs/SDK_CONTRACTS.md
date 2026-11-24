# SDK contracts and release boundary

NIL 0.2.1 remains alpha. The versioned [workflow inventory](WORKFLOWS.md) defines
bounded supported paths, independent of the broader 32-area roadmap. A stable
release requires compatibility tests against the declared Python/backend profiles,
packaged offline help, deterministic examples, explicit failure behavior and
migration notes for each changed persisted schema. This document specifies the
current boundary; it does not declare a stable 1.0 release.

| Family | Input | Result | Prediction | Persistence |
|---|---|---|---|---|
| Classical | estimator, features, optional targets/weights | ClassicalResult | nil.predict(result, features) | nil.save/load; trusted artifacts only |
| Neural | torch module, dataset/loader, task/config | TrainingResult | nil.predict(result, inputs, device_name=...) | nil.training.checkpoint |
| Online RL | native algorithm, environment | RLTrainingResult | nil.predict(result, observations) | nil.rl.save/load |
| Offline RL | declared transition dataset | OfflineResult | policy's native interface | offline workflow checkpoint |
| Centralized RL | per-agent transitions/policies | CentralizedResult | explicit policy mapping | centralized workflow checkpoint |
| Native | bounded native model/tensors | model/history/steps mapping | model call | native JSON save/load APIs |

`nil.contracts.result_info(result).to_dict()` gives a schema-1 metadata view of all
six families. Numerical payloads stay in their original outcomes. Paths are
references and may no longer exist. `MapDataset`, `Predictor`, and
`SingleModelResult` are structural typing protocols, not backend or shape checks.

Neural prediction/evaluation accept either a module or its TrainingResult and
restore its train/eval mode even after forward failure. They may move the model
to the selected device. Classical transforms use nil.transform and require an
estimator exposing transform. No callback, file or registration runs on import.

TrainingConfig validates precision, worker budgets and optimizer parameters.
Neural callbacks receive `(trainer, run, row)`; backend RL callbacks retain their
native contract. Neural cancellation is cooperative between batches, with an
explicit cancelled result; process execution limits provide separate containment.
Native fit and classical estimator fitting do not promise universal mid-fit
cancellation. Progress and lifecycle details remain family-specific.

Public helpers use ValueError for invalid declared options/data, TypeError for
unsupported object kinds, MissingDependencyError (an ImportError) for missing optional backends (with install
instructions), and native filesystem/backend exceptions where documented. Backend
exceptions are not erased or made falsely interchangeable. No global configuration
or environment mutation is required to use the structural interfaces.

Extensions use the existing process-local `nil.register_model(name, factory,
group=..., help_path=...)`. Factories load lazily, names cannot replace built-ins,
and a help.md accompanies each shipped module. Registration does not install
packages, load plugins from the network or introduce a second registry.

Independent-engine milestones remain separate: native dtype/device expansion,
custom/higher-order differentiation, broader CPU kernels, then actual accelerator
operators with numerical and resource acceptance. Integrated Torch/sklearn
coverage does not complete these milestones. GPU and actual older-PC acceptance
remain open in the [roadmap](project/TODO.md).

Recorded [2026-11-24 framework acceptance](benchmarks/2026-11-24-framework.json):
regular-wheel 153 Torch, 36 RL, 70 current-classical and 68 minimum-classical tests,
all with zero skips and all installed examples passing. Hosted full SDK regression
passed 1,253 tests with 46 explicitly skipped optional-target cases. Passive discovery
loaded no numerical backends; five fresh-process samples measured median NIL import
0.22ms (excluding interpreter startup), explicit inventory/package-metadata query
42.48ms, and at most 20,258,816-byte total interpreter peak RSS on this host.
These timings are observations, not performance guarantees or older-PC acceptance.

## Seven-workflow v1 contracts

The graph, contrastive, audio and fused-feature examples use ordinary Torch modules
and the existing nil.train/TrainingResult/checkpoint interface. Self-training returns
a native sklearn estimator and uses existing classical prediction/persistence.
Retrieval, tool plans, causal estimates and conjugate posteriors have explicit
module-specific reports/state; call a posterior's predict method directly rather
than assuming generic nil.predict dispatches every probabilistic object. Immutable
index/posterior/graph/replay snapshots share the existing bounded JSONL integrity
protocol. Audio/visual processor metadata and labels are saved separately from
model weights and must match on reload. See [workflow contracts](SEVEN_WORKFLOWS.md)
for exact input schemas, budgets, missing modalities, causal assumptions and the
macOS active-job-recovery refusal boundary.

Advanced seven-area additions and still-open requirements: [advanced workflow phase](ADVANCED_WORKFLOWS.md). New CPU checks are separate from the earlier v1 hosted matrix.
