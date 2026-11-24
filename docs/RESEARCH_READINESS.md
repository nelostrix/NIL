# NIL research readiness: efficient learning and compact intelligence

Updated 2026-11-24. This is a readiness audit and research specification, not a
claim that a compact model already matches a larger model. The working program
name **Ultra-Compact Intelligence (UCI)** denotes our proposed experiments across
existing research areas; it is not a claim to have invented those areas.

## Objective and scope

Two related programs guide implementation:

1. **Efficient learning:** reduce measured time, resource use and operational work
   for a specified training task, including cross-computer execution and offload.
2. **Compact intelligence:** improve useful task capability per stored parameter
   while accounting for additional computation, external knowledge and tools.

A 10–20B model approaching a named 100B–1T baseline on a declared suite is a
research target. Matching every capability is not established or guaranteed.
Against 1T, 20B means 50 times fewer parameters (98% reduction); 10B means 100
times fewer (99% reduction). Neither arithmetic comparison establishes quality.
Report whether a reference is dense or mixture-of-experts, total and activated
parameters, precision, tools, context and inference-compute budget.

The base package must remain NIL-only. Backends, network workers, datasets,
weights, solvers and low-bit kernels stay explicit opt-ins. New experiments use
existing shallow SDK entry points; there is no requirement to implement all 32
areas before beginning a local recursive-model experiment.

## Current capability audit

“Available” below means an existing bounded implementation. “Smoke-validated”
means an executable small integration check. Neither means large-model readiness.

| Research requirement | Current evidence | Missing acceptance |
|---|---|---|
| Arbitrary differentiable architectures/objectives | Torch modules, custom scalar-loss steps and weighted accumulation | Broader structured outputs, per-token normalization and research artifact contracts |
| Shared/repeated attention blocks | Compact example; shared-gradient parity and depth-independent parameter count | Useful language/reasoning results, long-depth stability and depth-generalization |
| Architecture/checkpoint identity | Existing structural preflight; example records depth in extra_repr | Stable portable research architecture manifests and migrations |
| Resume and reproducibility | Model/optimizer/RNG/loader state; compact shared-model exact CPU resume test | Halting/cache/tool state, asynchronous workers and new loader/device combinations |
| Adaptive computation | Users can write loops in a custom Torch module | Learned halting, regularization, per-example/token budgets and dedicated acceptance |
| Distillation | Offline classification-logit recipe with CE plus temperature-scaled KL | General token/sequence/feature/process distillation; teacher version/cache contracts |
| External knowledge | Bounded BM25 retrieval/RAG, citations, updates/deletes and immutable reload; existing memory components | Learned embeddings/hybrid reranking and grounded-generation evaluation beyond v1 |
| Symbolic/tool reasoning | Approved schema-checked Tool plans, owned deadlines/cancellation, durable prefix recovery and shortest paths | Autonomous goal policy, solver families, task competence and external-effect reconciliation beyond v1 |
| Low-bit models | Scoped dynamic and weight-only INT8 | Native ternary/INT4 training/storage/kernels; low-bit optimizer and fidelity acceptance |
| Training measurement | Run metrics/timing, supported resource acceptance, CPU inference profile | Per-run training FLOPs, accelerator peak memory, energy and full-system cost accounting |
| Multi-computer training by address | Scoped same-host CPU DDP/FSDP only | Authenticated join/launcher, data/environment distribution and real multi-host recovery |
| Parameter sharing under distributed execution | Local shared-model recipe | Current scoped FSDP rejects tied parameters; shared Transformer also falls outside its strict module restrictions |
| RAM/SSD configurable offload | Existing checkpoints and activation recomputation | Actual bounded CPU/NVMe offload planner, storage lifecycle, performance and fault tests |
| Larger pretrained models | Pinned 134.5M checkpoint accepted on the current CPU host | Larger checkpoint/task/device-specific acceptance; not a 1B/10B/1T certification |
| Audio/video/graphs/probabilistic/other regimes | Seven bounded v1 workflows now implemented; see SEVEN_WORKFLOWS.md | Advanced modalities/regimes, pretrained encoders and broader real-data/hardware acceptance remain open |

The strict distributed reference rejects dropout module types even when a model
sets dropout probability to zero, and its FSDP path rejects aliased parameter
names. Repeatedly calling a single registered block is different from registering
one parameter under multiple names; do not assume those cases are interchangeable.
The current distributed feature adapter also converts inputs to floating point,
which is unsuitable for the compact recipe's integer embedding token IDs. The
recipe is therefore a **local CPU** recipe; do not pass it through that distributed
API and advertise support. Fix and test the distributed
contract before adding a supported distributed research recipe.

`nil.optimization.profile` measures a cloned CPU inference model's footprint and
latency. It does not currently measure training peak RAM, FLOPs or energy.

## Executable readiness check

```sh
# Select an environment with explicitly installed Torch/NIL; no downloads occur.
python -I nil-sdk/examples/compact_research.py --output /tmp/nil-compact-seed7
PYTHONPATH=nil-sdk python -m pytest nil-sdk/tests/test_compact_research.py -q
```

The example trains an untied attention baseline, a repeated shared-block student,
and a student using frozen offline teacher logits. Each uses four attention
applications at width 16, with deterministic synthetic token-label data, separate
96/48 train/validation samples and no external tools. It records unique model
parameter counts, held-out loss/accuracy, timing, dataset hash and exact checkpoint
reload. The CLI requires a fresh output directory. Seeds and epoch limits are
explicit; all default training is CPU FP32 with zero loader workers.

The objective is alpha * CE(labels, student) + (1-alpha) * T^2 *
KL(teacher || student), averaged over examples. Teacher logits are detached;
class indices are integer labels. Tests compare the KL calculation independently,
verify that teacher gradients stay absent, compare uneven accumulated updates
with one full batch, prove shared gradients equal the sum of untied reference
uses, and verify exact resume. A changed reuse depth is rejected before loading
weights even though shared-model parameter shapes remain unchanged.

This is a small integration smoke, not the proposed 1M-versus-10M experiment,
a reproduction of Universal Transformers/ACT, an adaptive-halting implementation,
a compute-matched study or evidence of teacher-level intelligence. Teacher
production is included in total example wall time; per-model timing also contains
validation/checkpoint/reload work. The example's Python entry-point/class identity
must match at reload; stable cross-entry-point architecture packaging remains open.
Full CLI invocation wall time and energy are not measured by its internal timer.

## Seven compact-intelligence experiment tracks

| Track | First experiment | Critical measurements and controls |
|---|---|---|
| Recursive computation | Untied Transformer versus shared-block Transformer | Unique parameter count, equal-data and equal-compute comparisons, gradient stability, unseen depth/length |
| Adaptive thinking | Fixed-depth versus learned bounded halting | Per-example/token work, halt mass/remainder, max-step enforcement, pondering penalty, calibration and tail latency |
| External knowledge | Same compact model with/without frozen retrieval | Retrieval failures, freshness, provenance, answer quality, index/embedding-model footprint and latency |
| Extreme low-bit | FP32/BF16 baseline versus explicit low-bit variant | Quality, packed bytes, scales/padding, actual kernels/latency and training state; no inference-bit inference about optimizer RAM |
| Neuro-symbolic | Same model with/without selected executable tools | Correctness, tool/call/time budgets, solver failures, recovery, full-system resource use |
| Distillation | Identical student with/without teacher supervision | Teacher data/version, temperature/loss normalization, teacher cost and no test leakage |
| Alternative architectures | Transformer against selected recurrent/state-space/graph/hybrid candidate | Comparable data/token/parameter/compute budgets, independent correctness and architecture-specific state recovery |

Reference work: [Universal Transformers](https://arxiv.org/abs/1807.03819),
[Adaptive Computation Time](https://arxiv.org/abs/1603.08983),
[BitNet b1.58](https://arxiv.org/abs/2402.17764),
[knowledge distillation](https://arxiv.org/abs/1503.02531),
[Tiny Recursive Models](https://arxiv.org/abs/2510.04871) and
[ZeRO-Infinity](https://arxiv.org/abs/2104.07857).
These motivate experiments; their results are not NIL results. Published compact
reasoning work must be included when selecting strong task-specific baselines;
success on a narrow puzzle suite does not establish general language-model
equivalence. Pin paper/code/model revisions in the actual experiment protocol.

Theoretical ternary entropy is log2(3), approximately 1.585 bits per weight.
A nominal 1.58-bit 20B weight payload is approximately 3.95 decimal GB. Real storage
includes packing, scales, padding, unquantized tensors and metadata; a simple
fixed-width ternary encoding uses 2 bits per symbol. Training also needs
activations, gradients and optimizer state. No current NIL INT8 path is a BitNet
training implementation, and reduced storage alone does not increase capability.

## Evaluation contract before capability claims

1. Select named baseline models and pinned revisions, task definitions, held-out
   data and pass floors before training. Keep training, tuning and final test
   sets separate, with provenance, contamination controls and disclosure of
   synthetic/teacher-generated data.
2. Report task scores individually and specify any aggregate normalization and
   chance floor. Include calibration/abstention, failures, output/context limits,
   length/depth generalization and repeated-seed uncertainty where applicable.
3. Count unique total and activated neural parameters, buffers, packed and live
   bytes. Count every model used at inference: retriever, router, teacher/verifier
   or tool model. Report sparse and mixture-of-experts counts explicitly.
4. Run both parameter-efficiency and compute-matched comparisons. Reusing a block
   ten times is additional computation, and using a teacher consumes resources.
   Record training tokens, updates, batch size, FLOP measurement method, precision,
   device/kernel versions, wall time, memory, communication, I/O and checkpoints.
5. Include full-system inference costs: model/index/tool storage, RAM/VRAM peaks,
   energy if measured, network/API usage, number of model/tool calls, throughput,
   median/tail latency, and cold/warm behavior. Mark unmeasured fields explicitly.
6. Predeclare multiple seeds, compute/storage limits and stop rules. Publish raw
   per-run outcomes, confidence/variation, negative results and recovery behavior.
   No task cherry-picking after looking at final test results.

S/N can be an exploratory “score per billion unique parameters” statistic within
a fixed suite. It is not a validated intelligence measure or the primary release
gate. Use minimum task-capability floors and a quality/resource Pareto comparison;
a tiny inaccurate model should not win merely because its denominator is small.
External knowledge and extra thinking must remain visible in that comparison.

## Progressive scale gates

| Gate | Student / reference target | Required evidence before moving on |
|---|---|---|
| Smoke | Thousands of parameters, synthetic | Gradients, actual updates, held-out loss reduction, checkpoint identity/recovery; no capability conclusion |
| E1 | Approximately 1M / 10M | Fixed local datasets and protocol, at least 3 seeds, held-out generalization and measured compute/memory; outcome may be negative |
| E2 | Approximately 10M / 100M | Replication on another task/data distribution with matched training and inference budget reports |
| E3 | Approximately 50M / 500M | Real language/task quality, reproducible training/data manifests and target hardware acceptance |
| E4 | Approximately 100M / 1B | Broader suite, reliability, total teacher/tool/retrieval costs and independent repeat |
| E5 | Approximately 1B / 10B | Provisioned hardware, distributed/offload acceptance if used, complete compute and recovery evidence |
| E6 | Approximately 10–20B / 100B–1T | Named dense/MoE baselines, explicit equivalence margins per task, full-system costs and independent replication |

These sizes are research targets, not delivery promises or a guarantee of equal
quality. Run the next gate only if its hardware and budget exist and the preceding
results justify it. A useful capacity result can stand even when it is slower;
a speed claim needs a time-to-quality result against the agreed baseline.

## Infrastructure and commercial scope

For cross-computer work, begin with independently scheduled experiments before
synchronous shared-model training. An IP address identifies an endpoint; a worker
still needs authorization, compatible environments, a data contract and reliable
checkpoint ownership. Measure network/compute/storage first and retain a local
fallback when distributing would be slower. Mixed hardware and unstable internet
are separate targets from a supported homogeneous wired LAN.

An offload design must declare separate GPU/RAM/SSD budgets, bounded transfer
buffers, pinned-memory ownership, read/write accounting, cleanup and crash recovery.
Disk is a slower storage tier, not physical RAM; arbitrary OS swap is not the
training offload implementation. Any larger-model capacity gain and any speed gain
are separate acceptance results.

Paid orchestration, deployment and support require measured customer value. Add
cost metering, quotas and failure/refund semantics only after an actual service or
supported installation is defined. Do not sell an unmeasured speedup or charge for
“creating RAM.” Track customer pilots separately from research benchmark outcomes.

## Work tracking

[RESEARCH_TODO.md](project/RESEARCH_TODO.md) is the ordered, checkable research and
infrastructure plan. The [32-area master](project/TODO.md) remains the broader SDK
backlog, including all seven previously requested workflow areas. A checked smoke
item never closes a whole research track, hardware target or framework area.


## Recorded readiness outcome — 2026-11-24

Current recipe acceptance is Linux/Python 3.12 with the versions in the report.
Prior platform/backend CI evidence predates this example and is not a new
minimum-backend or cross-platform certification. The installed Torch profile
now includes the new checks for subsequent CI runs.

The [three-seed CPU smoke](benchmarks/2026-11-24-compact-readiness.json) passed
with seeds 7, 11, 19 and 10 focused tests with zero skips. The untied baseline has
9,282 parameters; each shared model has 2,610. All three recipes reduced held-out
loss and reproduced predictions exactly after checkpoint reload. Shared-gradient
parity, uneven distillation accumulation and exact CPU resume passed separately.

Distillation did not outperform the shared supervised model on any tested seed;
that negative result is retained. No measured FLOP/energy/peak-training-memory
result, intelligence-equivalence result or large-model readiness follows from
this smoke. Full measured1M-versus 10M research remains R005/P2, not completed work.


The regular installed-wheel Torch profile also passed 163 tests with zero skips
and all five examples, including compact_research.py. The base-install verifier
again passed offline with dependency resolution enabled and only nil-sdk added.
The dated JSON records exact interpreter/backend versions and wheel hash. No
new minimum-backend or accelerator acceptance is inferred from this local run.

The [seven-workflow follow-on](SEVEN_WORKFLOWS.md) adds usable scoped primitives
for the previously missing areas. It does not change the compact-intelligence
smoke into a large-model experiment or complete its outstanding research gates.
