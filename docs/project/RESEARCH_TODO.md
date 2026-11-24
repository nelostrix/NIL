# NIL efficient-learning and compact-intelligence TODO

Updated 2026-11-24. Status is acceptance-based. Research hypotheses and hardware
requests remain open until measured evidence exists. This plan supplements the
[32-area master](TODO.md), never replaces its unchecked requirements. See the
[readiness audit, protocols and scope](../RESEARCH_READINESS.md).

## P0 — Establish the actual framework boundary

- [x] **R001** Audit the seven proposed compact-intelligence directions against
  actual SDK interfaces; document local support and distributed/profile gaps.
- [x] **R002** Define capacity versus speed, parameter count versus compute, dense
  versus MoE and native implementation versus optional integration distinctions.
- [x] **R003** Define experimental score/resource reporting and staged scale gates;
  do not describe 10–20B-to-1T capability equivalence as an implemented feature.
- [x] **R004** Run and record the small offline shared-model/distillation readiness
  recipe with seeds 7/11/19, real updates, reload and exact resume/gradient tests.
  [Recorded smoke evidence](../benchmarks/2026-11-24-compact-readiness.json): 10 tests
  passed; distillation did not improve on the ordinary shared student.
- [ ] **R005** Publish a first1M-versus 10M experiment specification with selected
  tasks, pinned data, baselines (including relevant published compact-recursive
  work), margins, seeds and a real resource budget.
- [ ] **R006** Complete research model/data/objective/config/error contracts and
  reusable type hints; preserve existing classical/neural/RL interfaces.
- [ ] **R007** Define portable architecture manifests including block sharing,
  recurrence depth, halting config, tokenizer/processors and schema migrations.
- [ ] **R008** Validate altered objective/teacher/data/architecture rejection before
  model, optimizer, RNG, cache or dataset state mutation on resume.

Exit: local experiments are repeatable and report their limits. Large-model or
multi-host readiness is not inferred from this gate.

## P1 — Reproducible measurement before speed or intelligence claims

- [ ] **R010** Add a versioned research run schema for model/data/code/environment,
  config, raw metrics, random seeds and immutable artifact lineage.
- [ ] **R011** Count unique/activated parameters, shared storage, buffers, packed
  quantized state and all neural components used in the inference system.
- [ ] **R012** Measure training steps/tokens, useful examples, FLOPs with documented
  method, wall time and optimizer work; expose missing measurements explicitly.
- [ ] **R013** Measure process-tree RAM, device allocated/reserved/peak memory,
  I/O and communication; hardware-specific limits require actual devices.
- [ ] **R014** Measure cold/warm inference throughput and median/tail latency with
  fixed context, output, recursion and tool-call budgets.
- [ ] **R015** Integrate energy measurement only on supported instruments/devices;
  track any estimate separately from observed energy and hardware provenance.
- [ ] **R016** Add per-task capability floors, uncertainty/calibration and failure
  reports; define aggregation before testing and retain per-task raw scores.
- [ ] **R017** Add parameter-matched and compute-matched comparisons, seed variation,
  data contamination controls and held-out length/distribution generalization.
- [ ] **R018** Report complete teacher/retrieval/tool/API/index/network costs and
  publish negative results; do not optimize only the score/parameter ratio.
- [ ] **R019** Build a release/research dashboard separating implemented, validated,
  failed, experimental, unmeasured and target-blocked combinations.

Exit: a smaller/slower system cannot be presented as a training-speed win, and a
small denominator alone cannot pass a capability threshold.

## P2 — Recursive computation and adaptive thinking

- [ ] **R020** Implement a versioned recursive attention architecture/recipe with
  explicit state injection, positional/depth encoding and bounded work controls.
- [ ] **R021** Compare shared versus untied blocks at fixed data and declared
  parameter/compute budgets; count all embedding/head parameters.
- [ ] **R022** Validate gradients, initialization, normalization, numerical stability,
  masks and memory at increasing recurrence depths and sequence lengths.
- [ ] **R023** Add learned halting with halt probabilities, residual mass, pondering
  regularization, forced termination and train/eval semantics.
- [ ] **R024** Add per-example/token work accounting and deterministic inference
  ceilings; test empty/completed sequences and batch-dependent halting behavior.
- [ ] **R025** Validate recurrent/halting state checkpoints, exact supported resume,
  cancellation and changed-config refusal before mutation.
- [ ] **R026** Evaluate fixed-depth, recursive and adaptive baselines on held-out
  algorithmic, language and reasoning tasks with at least 3 prespecified seeds.
- [ ] **R027** Publish E1 results, including failures and compute overhead; proceed
  to E2 only if evidence and available budget justify it.

Exit: the small readiness example alone does not establish a new architecture or
adaptive-halting implementation.

## P3 — Distillation, external knowledge and executable reasoning

- [ ] **R030** Generalize the offline-logit recipe into an explicit distillation
  contract: teacher identity, tokenizer alignment, masks and temperature.
- [ ] **R031** Add token/sequence/representation objectives, correct denominators,
  frozen teacher behavior and validated accumulation/precision combinations.
- [ ] **R032** Add bounded versioned teacher caches with provenance, checksums,
  resumable generation, stale-cache rejection and no evaluation leakage.
- [ ] **R033** Compare identical students with/without teacher supervision; count
  teacher training/generation/API cost and report quality retention per task.
- [ ] **R034** Complete retrieval/RAG ingestion, chunking, embedding/index/reranking
  and provenance contracts using existing memory components where suitable.
- [ ] **R035** Validate retrieval freshness, updates/deletes, missing/conflicting
  knowledge, access boundaries, citations and index restart/migration.
- [ ] **R036** Add grounded-answer/retrieval evaluation with fixed corpora and
  compact-model-alone versus retrieval-assisted comparisons.
- [ ] **R037** Complete agent/planning tool schemas, argument/result validation,
  permissions, budgets, deadlines, cancellation and restart/replay contracts.
- [ ] **R038** Add selected calculator/code/logic/search/solver adapters with typed
  errors and independently verified outcomes; test real execution boundaries.
- [ ] **R039** Compare tool-assisted versus tool-free models at explicit latency,
  call and resource budgets; track every external model and store.

Exit: document where capability comes from, including external components;
retrieval or tool access is not a claim of capability stored in student weights.

## P4 — Low-bit execution and alternative architectures

- [ ] **R040** Define ternary/low-bit forward/backward, scaling, rounding and
  optimizer/master-state semantics before training support is advertised.
- [ ] **R041** Implement reference low-bit operators and independently check
  gradients/surrogates, saturation and numerical error against FP baselines.
- [ ] **R042** Define packed storage, serialization/migration and metadata accounting;
  theoretical 1.58-bit entropy must not be reported as actual training RAM.
- [ ] **R043** Add real selected CPU/GPU low-bit kernels with correctness, shape,
  device and measured latency acceptance; no simulated integer speed claims.
- [ ] **R044** Compare from-scratch and adapted low-bit models under declared data,
  compute and fidelity gates; preserve failed quantization configurations.
- [ ] **R045** Add independently scoped recurrent/state-space/hierarchical/sparse
  architecture recipes without making all backends mandatory.
- [ ] **R046** Validate architecture-specific state, masks, gradients, checkpoint
  identity and streaming behavior before comparing capability.
- [ ] **R047** Compare selected alternatives against a strong Transformer baseline
  at equal declared data/compute and report task-specific limitations.

Exit: successful packing or a reference forward is not a useful training kernel
or proof of increased intelligence density.

## P5 — Cross-computer execution by address

- [ ] **R050** Define a supported initial topology/backend/OS set and a lightweight
  opt-in worker with explicit pairing/credentials, permissions and revocation.
- [ ] **R051** Add address-based discovery/enrollment plus version/hardware/network/
  storage preflight; no process/network listener starts on normal SDK import.
- [ ] **R052** Add owned remote job lifecycle, bounded logs, dependency manifests,
  authenticated artifact/data transfer and cleanup on cancel/disconnect.
- [ ] **R053** Distribute independent experiments/search/evaluation first, including
  leases, retries, idempotency, resource limits and committed result ownership.
- [ ] **R054** Add real multi-host training launch/rendezvous, rank/device mapping,
  data sharding, deterministic seeds and compatible backend selection.
- [ ] **R055** Support shared/tied parameters and chosen recursive architectures in
  selected distributed strategies; current scoped FSDP rejection remains a gate.
  Add integer token IDs and structured batch support rather than converting all
  input features to float; test repeated blocks separately from aliased names.
- [ ] **R056** Implement shard/checkpoint ownership, coordinated failure handling,
  interruption/rejoin and tested continuation at explicitly defined boundaries.
- [ ] **R057** Measure compute/communication balance and recommend local/distributed
  placement; an extra slow worker must not imply an automatic speedup.
- [ ] **R058** Validate on at least two actual machines; keep same-host multi-process
  tests separate from real network, firewall and host-failure acceptance.
- [ ] **R059** Scope heterogeneous hardware/internet operation as separate later
  targets; measure stragglers, bottlenecks and data consistency before release.

Exit: entering IP addresses is a usable interface backed by actual lifecycle and
security contracts, not a claim that IP addresses alone establish a cluster.

## P6 — Configurable GPU/RAM/SSD budgets and offload

- [ ] **R060** Profile model state, activations, optimizer, data and transfer buffers;
  define independent physical-memory and storage budgets with safety headroom.
- [ ] **R061** Define explicit memory tiers and bounded CPU/NVMe offload adapters;
  never silently enable whole-system swap or describe disk as new physical RAM.
- [ ] **R062** Add asynchronous bounded transfer/prefetch, pinned-memory accounting,
  synchronization and allocator ownership with numerical correctness tests.
- [ ] **R063** Add offload-file locking, integrity, durable checkpoints, disk-full/
  corrupt/read-failure behavior, cancellation and owned scratch cleanup.
- [ ] **R064** Compare in-memory, checkpointed and offloaded training for identical
  results, model capacity, wall time, transfer volume and peak live resources.
- [ ] **R065** Measure actual storage device throughput/endurance implications and
  contention; reject unsafe budgets rather than promise arbitrary capacity.
- [ ] **R066** Accept selected offload configurations on actual target hardware;
  distinguish larger-model feasibility from lower time-to-quality or cost.
- [ ] **R067** Combine distributed/offload only after each contract passes, then
  test aggregate device/host/storage limits and worker-loss recovery.

Exit: no universal speed claim, no “SSD becomes RAM” claim and no configuration-
only acceptance for hardware we do not have.

## P7 — Complete the previously requested missing workflow areas

All seven now have implemented v1 workflows, with their checked deliverables in
the master areas and [scope/evidence guide](../SEVEN_WORKFLOWS.md). The broader
completion items below retain advanced requirements beyond v1. They support future
experiments but do not all block P0–P2. Keep detailed algorithm requirements in their master-area checklists.

- [ ] **R070** Complete area14 retrieval/RAG workflows and acceptance (R034–R036).
- [ ] **R071** Complete area20 agents/planning execution and acceptance (R037–R039).
- [ ] **R072** Complete area22 probabilistic/causal workflows: selected inference,
  identification assumptions, diagnostics and statistical reference validation.
- [ ] **R073** Complete area23 self/semi-supervised, active, few-shot and continual
  recipes, including held-out data efficiency/forgetting and state recovery.
- [ ] **R074** Complete area16 speech/audio waveform/processors, selected tasks,
  streaming boundaries, metrics, artifacts and real-data acceptance.
- [ ] **R075** Complete area17 video/multimodal sampling/alignment/processors,
  selected tasks, absent/corrupt modalities, artifacts and real-data acceptance.
- [ ] **R076** Complete area21 graph storage/sampling/batching, selected message-
  passing/node/edge/graph tasks, leakage-safe splits and real-data acceptance.

Exit: adding a custom-model example or backend alias does not complete a modality.

## P8 — Scale, packaging and customer value

- [ ] **R080** Run E2–E4 only after lower gates, data access and compute budgets are
  accepted; publish all task/seed and negative results.
- [ ] **R081** Provision real GPU/multi-host/older-PC targets and selected larger
  checkpoints; current host constraints remain unchanged until supplied.
- [ ] **R082** Run E5/E6 with named baseline revisions, total/active parameter counts,
  per-task equivalence margins and full-system compute/resource accounting.
- [ ] **R083** Independently replicate meaningful improvements; publish configs,
  code/data provenance, checkpoints where distributable and reproduction commands.
- [ ] **R084** Verify NIL-only base wheels, lazy imports, offline help/examples,
  source/wheel parity, supported Python/backend matrix and artifact migrations.
- [ ] **R085** Add bounded research modules and help only when reusable interfaces
  are justified; avoid seven separate overlapping engines or mandatory extras.
- [ ] **R086** Validate customer workflows through interviews and paid pilots;
  measure engineering time, task quality, reliability and total operating cost.
- [ ] **R087** Define paid orchestration/support or supplied-compute plans only
  against demonstrated value, service ownership, metering and failure policy.
- [ ] **R088** Keep a dated public scope/results report: no unsupported “1T-level,”
  universal acceleration or zero-overhead promises based on small smoke tests.

## Immediate execution order

R004 → R005–R008 → P1 measurement → P2 recursive/adaptive study. Run independent
job orchestration (P5) and offload design (P6) when their target machines exist.
P3/P4 and the seven missing workflow areas progress through explicit scoped gates;
P8 scale requires prior evidence and resources. This ordering can be revised from
actual experiments and customer feedback, with the revision recorded explicitly.
