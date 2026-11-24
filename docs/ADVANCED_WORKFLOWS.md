# Advanced workflow phase

Subsequent work: [chunked speech, continual training and heterogeneous graphs](CONTINUED_WORKFLOWS.md). This document records the earlier phase and its historical open scope.

This phase extends the seven [foundation workflows](SEVEN_WORKFLOWS.md).
The framework remains alpha. These are bounded implementations, not completion
of seven entire AI fields. Base installation still installs **only nil-sdk**;
no model, codec, backend or service starts on installation or import.

| Area | Added in this phase | Still open |
|---|---|---|
| Retrieval/RAG | Atomic bulk ingestion; legacy snapshot migration; local mean-pooled Transformer encoding; cosine and hybrid reciprocal-rank retrieval; explicit reranker callback; reference answer EM/F1 and document citation precision | Learned reranker training; semantic entailment/faithfulness; database-scale indexes and migrations; production ACL and multi-writer ingestion |
| Agents/planning | Bounded policy-driven tool loop with approved tools, per-call deadlines, version-bound journals and committed-call resume; explicit verified-result reconciliation | General autonomous reasoning; automatic external-effect reconciliation; concurrent journal ownership; cross-service transactions |
| Probabilistic/causal | Multi-chain random-walk Metropolis, exact state/RNG continuation, basic split-Rhat/ESS; linear IV 2SLS with HC1 uncertainty and first-stage diagnostic | HMC/NUTS, full probabilistic programming, causal discovery, longitudinal/time-varying treatment identification |
| Learning regimes | Functional first/second-order MAML inner adaptation; measured continual-learning accuracy/forgetting summaries | General episodic task engine; resumable end-to-end continual-learning trainer; large-task meta-learning and rigorous representation benchmarks |
| Audio/speech | Explicit local native CTC ASR, experimental seq2seq loader and VITS TTS; mono conversion, exclusive PCM16 output, WER, optional anti-alias polyphase resampling | Streaming/diarization; multilingual quality acceptance; ASR/TTS fine-tuning workflows; hardware acceleration acceptance |
| Video/multimodal | Local native CLIP text/image embeddings and reload; optional original video frame timestamps/durations | Semantic video/audio-language training; live capture; model-family/codec coverage and real-world retrieval quality |
| Graphs | Weighted/directed GCN, row/symmetric propagation; true-nonedge sampling, held-out link splits, dot-product links and bounded CPU neighborhood sampling | Heterogeneous/temporal graphs, attention/message-passing families, scalable distributed graph loaders |

## Installation and local model loading

Keep `pip install nil-sdk` for the backend-free package. Explicit extras select
backends: `nil-sdk[classical]` for NumPy/scikit-learn workflows,
`nil-sdk[torch]` for graph/meta-learning, `nil-sdk[signal]` for polyphase audio,
and `nil-sdk[generative,vision]` for these Transformers speech/CLIP adapters.
These extras do install their declared dependencies. Backends are never mirrored
or copied into NIL's base wheel.

Adapters accept an **existing local model directory**, never a model repository
ID. They call native Transformers with `local_files_only=True` and
`trust_remote_code=False`. Current accepted adapters require CPU parameters.
Directory hashing is streaming and bounded (default 1 GiB, at most 128 selected
root files). Quantization, sharded architectures and arbitrary custom model code
are not implied. Model loading can still consume substantial RAM: a small SDK
does not make large weights or their computation small.

```python
import nil
encoder = nil.rag.load_encoder('/models/local-text-model')
index = nil.rag.semantic_index(encoder, mode='hybrid')
index.add_many([
    {'id': 'manual', 'text': 'NIL has explicit local model workflows.'},
    {'id': 'hardware', 'text': 'CPU and GPU acceptance are measured separately.'},
])
print(index.ask('How are local models handled?'))
artifact = index.save('/artifacts/new-index')
restored = type(index).load(artifact, encoder=encoder)
```

A supplied encoder needs `encode(texts)` and a stable `fingerprint`. Normalized
float32 embeddings are limited to one million stored elements. Reload validates
encoder/reranker identities and embedding digests without re-encoding documents.
A custom reranker receives `(question, candidate_dicts)` and must return one finite
score per candidate; provide its explicit `reranker_id` again on reload.
Do not mutate encoder weights/tokenization while its index is in use; explicitly
rebuild after changes. Ranking and filters precede answer generation. Answer
citation validation is syntactic; EM/F1 and document precision do not establish
that a statement follows from the cited passage.

## Planning and learning contracts

`nil.planning.agent(goal, policy, tools, policy_version=..., allowed=...,
 tool_versions=..., journal_root=..., max_calls=8)` uses an importable `nil.Tool`
as its policy. The policy accepts `goal` and `history` and returns either
`{'tool': name, 'arguments': {...}}` or `{'final': json_value}`. Each decision and
action runs in an owned process, through the same schema/permission checks as
`planning.run`. Deadlines apply per call, not to the complete loop. Exhaustion
returns `budget_exhausted`. Resume requires the identical contract; completed
calls are reused. A policy version is a caller assertion, not source attestation.

Failed calls are never automatically retried. `planning.reconcile(journal,
step_id=..., result=..., evidence=...)` accepts a caller-verified output for a
**failed** active step and records the evidence. It refuses a `running` journal;
recover/terminate its owned job first. The caller must verify external effects
and ensure exclusive journal ownership. No concurrent writers or distributed
exactly-once guarantee is claimed. macOS active-job identity recovery continues
to refuse unsupported operation safely.

`nil.learning.adapt(model, support_x, support_y, loss, steps=1, lr=.01)` returns
adapted parameters for `torch.func.functional_call`. It does not mutate model
parameters or existing gradients. Default second-order differentiation connects
query loss to the original parameters; `first_order=True` detaches inner gradients.
Use a deterministic eval-mode, buffer-free module with all parameters trainable;
this phase caps inner steps at 20 and parameters at one million. Outer optimization,
episode creation, test isolation and checkpointing remain caller-owned.

`nil.learning.continual_metrics(matrix)` requires a square stage-by-task matrix,
with `None` for future tasks. It reports final average, per-task forgetting,
mean forgetting and backward transfer. Negative forgetting means improvement.
ReplayMemory remains the separate explicit bounded reservoir, not an automatic
continual-learning algorithm.

## Sampling, causal inference and graphs

`nil.probabilistic.metropolis(log_density, initial, target_id='...')` uses
2–32 chains. Call `.sample(draws, warmup=..., thin=...)`, `.save(new_path)` and
`nil.probabilistic.load_metropolis(..., log_density=..., target_id=...)`.
Callbacks must be pure/deterministic; invalid density leaves sampler/RNG state
unchanged. Basic split-Rhat and autocorrelation ESS are diagnostics, not proof
of convergence or substitutes for rank-normalized diagnostics.

`instrumental_ate(z, treatment, outcome, controls=...)` fits linear 2SLS with an
intercept, HC1 heteroskedastic standard errors and an asymptotic normal interval.
Instrument relevance, exogeneity, exclusion, independent observations and the
linear constant-effect assumption are caller responsibilities. A high first-stage
F statistic cannot establish validity; weak-instrument-robust intervals are open.

`weighted_propagate(x, edges, weights, directed=..., normalization='row')` sums
duplicate COO edges. In undirected mode supply each unordered edge once; existing
self-loop weights are preserved, missing loops receive weight 1. Row normalization
uses incoming degree; symmetric normalization uses source-outgoing and target-incoming
degree. `weighted_gcn` reuses existing node/graph training contracts.
`split_links` excludes **all** supplied positives from negative samples, including
held-out links. Use train-positive edges alone for message passing. `negative_edges`
samples the exact complement without dense adjacency or unbounded rejection.
`sample_neighbors` is bounded CPU sampling, not an unbiased full-GCN estimator.

## Media contracts and reproducible validation

```python
asr = nil.audio.load_asr('/models/local-ctc')  # kind='seq2seq' is explicit
wave = nil.audio.mono(nil.audio.load_wav('/data/recording.wav'))
wave = nil.audio.resample(wave, asr.sample_rate, method='polyphase')
print(asr.transcribe(wave)['text'])  # explicitly segment inputs longer than 30 s

tts = nil.audio.load_tts('/models/local-vits')
spoken = tts.synthesize('An explicit local workflow.', seed=42)
nil.audio.save_wav(spoken, '/artifacts/new-speech.wav')

clip = nil.multimodal.load_video('/data/input.mp4', preserve_timestamps=True)
encoder = nil.multimodal.load_encoder('/models/local-clip')
# encode_images accepts bounded PIL images; encode_text returns normalized vectors.
```

Polyphase filtering needs SciPy and can overshoot input peaks. Normalize input
explicitly if needed; NIL refuses out-of-range output rather than silently clipping.
TTS preserves caller CPU RNG state and model mode. Adapters save native model plus
processor into a new directory with an integrity manifest, and refuse overwrite.
WER case-folds Unicode words and removes punctuation, with at most 2,000 words.

`preserve_timestamps=True` uses FFmpeg passthrough frames and bounded ffprobe frame
metadata; it preserves source timestamps rather than inventing a constant grid.
Missing/negative/nonmonotonic timestamps or missing last-frame duration are refused.
The default remains explicit CFR sampling. FFmpeg/ffprobe must already be installed.

Tests use independent closed-form references for MAML, 2SLS and weighted graph
operators; exhaustive small-graph complements; actual owned agent calls/resume;
Gaussian sampling moments/restart/failure rollback; anti-alias attenuation and WAV
roundtrip; original VFR timestamps versus an independent probe. Tiny local native
BERT/CTC/CLIP fixtures test model execution and exact reload without downloads.

`scripts/prepare_advanced_models.py --output /external/models` is an explicit,
separate download utility; pinned revisions and digests are recorded. The offline
`scripts/accept_advanced_models.py` accepts those directories and an explicit WAV.
Recorded pretrained probes establish execution and reload, not general accuracy:
ASR fixture has no gold transcript (WER unmeasured); TTS intelligibility is unmeasured;
vision uses synthetic colors. The MMS English TTS research fixture has
[CC-BY-NC-4.0 weights](https://huggingface.co/facebook/mms-tts-eng), unsuitable as
a commercial-weight entitlement. No weights or recordings ship with NIL.

## Final acceptance and remaining work

[Raw results, pinned model manifests and failure history](benchmarks/2026-11-24-advanced-workflows.json)
record the scope of every gate. Verification commit:
`9f78f3e4fdc7eb2a6e5ddbfd1012c2bb1c628c40` on the approved temporary branch.

| Gate | Result |
|---|---|
| Full hosted SDK regression | 1,321 passed, 46 optional skips, 105.85 s |
| Final local full regression | 1,198 passed, 32 optional skips, 125.77 s |
| Base platform/Python matrix | Nine Linux/macOS/Windows × Python3.10–3.12 jobs passed |
| Installed backend matrix | Ten minimum/current jobs passed; zero skipped acceptance tests |
| Final installed-wheel advanced subset | 15 passed, zero skips, 6.19 s, outside checkout |
| Installed pretrained probes | Four native models executed and reloaded offline; 21.82 s total |
| Pretrained resource measurement | Linux one-process lifetime peak899,552KiB; not a process-tree/quota gate |
| Final base wheel | 592,133 bytes; 223 Python files; 114 offline help guides; no mandatory dependencies |

The CLIP reload gate uses absolute/relative tolerance1e-6; observed maximum absolute
error2.9802322e-8. Other recorded probes compare exact outputs where stated. Earlier
failed probes and the hosted Windows status-file sharing race are retained in the
JSON evidence. The Windows fix retries sharing/access errors for at most0.5s;
persistent denial and corrupt JSON still fail. The previously failing minimum
Windows profile and all three Windows Python base jobs passed after the fix.

Native save_pretrained may rewrite configuration files and change the conservative
content fingerprint. Save and reload the canonical encoder bundle before building
an index intended to travel with it.

The table at the start lists the open work in each of the seven areas. Native
seq2seq ASR loading is exposed experimentally: CTC recognition executed here,
not a seq2seq recognition-quality gate. Real transcript-based WER, TTS intelligibility,
representative multimodal quality and broader graph benchmarks remain open.
General autonomous agents, end-to-end continual learning, probabilistic programming,
heterogeneous graphs and scalable multi-host training are separate future phases.
Actual GPU/older-PC targets remain unavailable. The compact-intelligence research
program is still in [its dedicated backlog](project/RESEARCH_TODO.md); no
10–20B-to-1T capability equivalence is claimed.
