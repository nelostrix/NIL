# Seven new AI workflow areas: implemented v1 scope

Updated 2026-11-24. The seven areas now have runnable, bounded workflows rather
than only generic custom-model foundations. These implementations do not close
every advanced requirement in the 32-area roadmap. Each new module is shallow,
loads numerical backends only when called and has bundled offline help.

| Area | Public module and implemented workflow | Explicit remaining scope |
|---|---|---|
|14 Retrieval/RAG | nil.rag: BM25 text chunks, source/offset/hash citations, access filtering, update/delete, immutable index reload, retrieval metrics, extractive answers or explicit generator callback | Dense embeddings, hybrid semantic retrieval/reranking, general document parsers, grounded-generation evaluation and automatic index migrations |
|20 Agents/planning | nil.planning: existing Tool schemas, approved ordered plans, prior-result references, owned tool processes/deadlines, cancellation, durable traces, safe-prefix resume and shortest-path search | General autonomous goal planning, LLM policy quality, distributed execution and external-side-effect reconciliation automation |
|22 Probability/causality | nil.probabilistic: conjugate Beta-Bernoulli, Gaussian Bayesian linear regression/prediction, cross-fitted AIPW binary-treatment ATE | General probabilistic programming/MCMC, causal graph discovery, instruments, longitudinal treatments and unmeasured-confounding identification |
|23 Other regimes | nil.learning: entropy pool queries, sklearn self-training, two-view NT-Xent, few-shot prototypes, bounded resumable reservoir replay | General meta-learning, large representation recipes, robust active-learning gains and measured continual forgetting/transfer |
|16 Audio | nil.audio: bounded 8/16/24/32-bit PCM WAV loading, channels/rates, explicit linear resampling, segments and incremental spectral classification features | Speech recognition/synthesis, speaker recognition, compressed audio codecs, bandlimited resampling and broad acoustic-model quality |
|17 Video/multimodal | nil.multimodal: GIF/TIFF timing, opt-in owned FFmpeg MP4/WebM decode, sampled timestamps, audio alignment, visual summaries and fixed-schema feature fusion with presence masks | Pretrained multimodal processors, semantic video tasks, ASR/video generation, original variable-frame timestamp preservation and live capture |
|21 Graph learning | nil.graphs: validated homogeneous undirected graphs, sparse normalized propagation, disjoint batching, node/graph GCN classification, pooling, JSON graphs and neural checkpoints | Directed/weighted/heterogeneous graphs, attention/message-passing families, neighbor sampling, link prediction and broad graph datasets |

## Installation and examples

Base install remains `pip install .` with zero mandatory dependencies. Choose
only the numerical backends your workflow uses:

```sh
python -m pip install '.[classical]'       # Statistics, causal ATE, active/semi/few-shot
python -m pip install '.[audio]'           # NumPy PCM/spectral features
python -m pip install '.[torch,vision]'    # Neural graph/representation/media examples

python nil-sdk/examples/knowledge_workflows.py --output /tmp/nil-knowledge
python nil-sdk/examples/statistical_workflows.py --output /tmp/nil-statistics
python nil-sdk/examples/graph_workflows.py --output /tmp/nil-graphs
python nil-sdk/examples/media_workflows.py --output /tmp/nil-media
python nil-sdk/examples/representation_workflows.py --output /tmp/nil-representation
```

Each command uses a fresh output directory. They train/save/reload where relevant.
They never download models or datasets. The statistical example uses sklearn's
bundled real Iris dataset; synthetic fixtures are identified explicitly elsewhere.
FFmpeg/ffprobe are external opt-ins for load_video; NIL does not install them.
No additional optional package is needed for the default GIF/TIFF example beyond
vision. Graphs do not require PyG/DGL/NetworkX at runtime.

## Retrieval, provenance and generation

```python
import nil
index = nil.rag.Index(chunk_chars=1000, overlap=100)
index.add('manual', 'The pump operating temperature is 40 degrees.', source='manual.txt')
answer = index.ask('pump temperature')
print(answer['text'], answer['citations'])
index.save('/tmp/manual-index')
restored = nil.rag.Index.load('/tmp/manual-index')
```

Indexing tokenizes Unicode words case-insensitively. Ranking uses BM25 term
frequency/document frequency/length normalization, with deterministic ID tie
breaking. This is genuine lexical retrieval; no hash vector is represented as a
learned embedding. Chunks retain source, document ID, character offsets and full-
document SHA-256. Replacing/deleting a document rebuilds its index transactionally
in memory; a budget failure leaves the previous content available. Rebuild/load
is explicit, without background services or an external vector database.

Default limits: 1000 documents, 8 million text characters, 10000 chunks, query length
10000 characters and k<=100. Index snapshots use the existing integrity-checked,
exclusive immutable JSONL protocol (16MiB artifact ceiling). Do not overwrite an
existing snapshot; publish another version. Document updates preserve only latest
content; caller-owned old snapshots provide history. Automatic schema migration
and streaming source adapters are not implemented.

`allowed_documents` filters results before generation; it is a caller-supplied
allowlist, not an identity provider or document-level tenant isolation system.
Without matches ask abstains. Default answers quote retrieved text. An explicit
`generator(question,hits)` callback must return text and citations to supplied
chunk IDs; syntactic citation validation cannot prove that generated claims are
faithful. External model calls and generation costs remain explicit/caller-owned.
`evaluate` reports document-level recall@k and MRR, not answer truthfulness.

## Tool planning and recovery

Plans contain ordered `{id,tool,arguments}` records. Existing Tool schemas validate
resolved arguments; names must be in the supplied tools/allowed/version mappings.
An argument `{'$ref':'earlier_step'}` references an entire prior JSON result.
Future/unknown references, duplicate IDs and denied tools fail before calls run.
`shortest_path` separately finds an unweighted path through explicit string states.

Each tool executes through NIL's existing owned process-tree executor. The timeout
is per call. Results are limited to 16KiB JSON and journals to 1MiB; default 32 steps,
maximum 1000. Importable decorated tools and picklable Tool objects are supported;
closures and arbitrary local classes are not worker-transfer contracts. Tool code
is trusted local code running with the user's privileges, not a hostile-code sandbox.

The journal records `running` before a call and commits its result afterward.
Resume reuses a completed prefix only when plan, permissions, versions and timeout
match. Completed calls are not repeated. A crash after a side effect but before
commit is ambiguous: failed/running steps require caller reconciliation and are
never retried automatically. There is no exactly-once external-side-effect claim.
Versions are explicit caller assertions; pin executable tool code with your run
artifacts. Do not edit journals while a runner owns them or run simultaneous owners.
A cancellation before a call can resume; cancellation during a call fails that step
and requires the same reconciliation as other interrupted external actions.

## Probability and causal assumptions

BetaBernoulli validates binary observations and updates alpha/beta only after the
entire bounded batch passes. It returns exact posterior mean/variance. Gaussian
BayesianLinear assumes fixed known observation variance and an isotropic Gaussian
weight prior; add an intercept column explicitly. Prediction includes observation
noise in its standard deviation. JSON posterior snapshots preserve predictions
without executable pickle. Feature dimension is bounded to 512.

`causal_ate(X,t,y)` cross-fits treatment probabilities and two outcome regressions
using stratified folds; callers' estimators are cloned. Inputs are independent rows,
binary treatment and finite continuous outcomes. Propensity overlap violations fail
rather than silently clip. The estimate is the average AIPW pseudo-outcome; the
interval uses the asymptotic influence-function standard error/normal approximation.
It assumes consistency, positivity and no unmeasured confounding given features.
Data and estimator correctness checks cannot establish those causal assumptions.
Intervals are not finite-sample, clustered-data or post-selection guarantees.
No causal conclusion follows from prediction accuracy alone.

See [doubly robust estimation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3070495/).
The example uses a transparent simulated known effect; it is not an observational
policy or clinical validation. Reports are plain JSON-compatible dictionaries.

## Other learning regimes

- Active queries rank predictive entropy, exclude previously labeled pool indices
  and enforce a label count budget. The oracle, label provenance and model refit
  remain explicit. Uncertain sampling is not guaranteed to improve held-out quality.
- Semi-supervised self_training clones an explicit sklearn estimator and uses -1
  for unlabeled rows, a threshold and finite iteration budget. Its native fitted
  estimator uses existing nil.save/load with the usual trusted artifact boundary.
- contrastive_loss implements symmetric two-view NT-Xent with self-similarity
  excluded. It returns a scalar differentiable Torch loss for custom nil.train
  steps. At most512 paired rows are accepted. Gradient accumulation does not create
  a larger shared negative pool; augmentation identity/correctness is caller-owned.
- prototypes uses only support embeddings/labels to form class means and classify
  independent query embeddings. Save support data with immutable snapshots. It
  does not train a language-model few-shot policy or a general meta-learner.
- ReplayMemory is bounded deterministic reservoir sampling over JSON records,
  with seen count/seed and exclusive JSON snapshots for exact continuation.
  A record is at most 4KiB; capacity is a count bound, not a hard process quota.
  Replay alone is not evidence of reduced forgetting or atomic model/data recovery.

References: [sklearn self-training](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html),
[SimCLR/NT-Xent](https://arxiv.org/abs/2002.05709). The implementation is not a
reproduction of published ImageNet results. Existing LoRA remains the separate
transfer/adaptation integration.

## Audio/video and multimodal contracts

WAV samples are float32 `[channels,samples]` in [-1,1]. Only uncompressed integer
PCM is supported, up to 8 channels/192kHz/60 seconds and a default 32MiB actual read
budget. Truncated data fails. Resampling uses explicit linear interpolation;
it is not a bandlimited signal-processing guarantee. Spectral features are
channel-mean Hann-window log-power band summaries (mean/std), RMS and zero-crossing
rate. Persist sample rate and FFT/hop/band configuration with the classifier.
The frontend stores running STFT summaries rather than a full spectrogram.

GIF/TIFF decoding preserves supplied frame durations; missing duration uses an
explicit default. Frames are white-alpha-composited RGB float32 TCHW. Per-frame
source pixels and total decoded bytes are bounded. Clip frame-count overflow
fails instead of returning an undisclosed truncated clip. Raw decoded tensors,
intermediates and stacking still consume memory; these limits are not OS quotas.

`load_video` explicitly invokes installed ffprobe/ffmpeg through an owned worker.
Only local file/pipe input protocols are enabled. Output is resized RGB on a
constant-rate fps grid starting at zero; original variable-frame timestamps and
audio are not silently preserved. Default source limit 64MiB, source pixels 4Mi,
frames 64, decoded float payload 64MiB, per-job timeout 30 seconds. Source files are
trusted local inputs and may change after stat; this is not a hostile-file sandbox.
FFmpeg executable/codec versions are part of reproducibility. GPU decoding is not
claimed. Missing executables fail with installation guidance, not an auto-install.

aligned_audio rejects gaps/insufficient coverage instead of padding silently.
fuse_features sorts declared names, checks fixed vector dimensions and appends
presence bits. Missing modalities fail unless zeros is explicitly selected. A
text encoder can supply a vector; no text/vision/audio encoder is downloaded.
The processor manifest, class order and model architecture must match at inference.
The media example demonstrates audio-only and fused classification plus exact
weight reload; generated tones/colors are wiring/codec tests, not speech or
natural-video intelligence results.

## Graph representation and training

Features are finite float32/float64 `[nodes,features]`; edges are int64 `[2,edges]`
on the same device. Inputs are bounded to 100000 nodes,1 million edges and 4 million
feature elements. The model has an explicit parameter budget. Undirected edge
pairs are symmetrized/deduplicated; each node receives exactly one self loop.
Propagation is D^-1/2(A+I)D^-1/2 X, with no dense adjacency allocation.
Batching offsets nodes to keep graphs disjoint; pooling averages contiguous graph
IDs. GCN uses two trainable linear layers, ReLU and normalized propagation;
`task='node'` returns node logits, `task='graph'` mean-pools graph logits.

Graph JSON snapshots preserve supported feature dtype and edges; label/split
ownership is explicit. Models use the existing structural neural checkpoints.
Node examples are transductive: held-out topology/features are visible, held-out
labels are excluded from the training loss. This differs from unseen-graph
classification, which is exercised separately in tests. Independent dense-matrix
normalization/gradient references and cross-graph noninteraction tests pass.

Reference: [Kipf/Welling GCN](https://arxiv.org/abs/1609.02907). No PyG/DGL backend
or published citation-network accuracy is implied. A separate real karate-club
run records 34 nodes/78 edges, 2 labeled training nodes and 32 held-out nodes; its
community labels/topology and dataset source must accompany the reported result.

## Validation and evidence

Focused tests cover independent BM25/posterior/KL/GCN references, actual updates,
held-out evaluation, immutable persistence, exact reload/replay continuation,
permissions, cancellation/deadlines, malformed/corrupt inputs, missing modalities,
codec limits and graph splitting. Installed profiles copy tests/examples outside
the checkout and import the regular wheel. Missing optional backends never count
as passing that backend's acceptance.

The evidence is [2026-11-24-seven-workflows.json](benchmarks/2026-11-24-seven-workflows.json).
It distinguishes real Iris/karate/recorded-audio fixtures from synthetic controls,
source versus installed acceptance, and external FFmpeg acceptance. Downloaded
development fixtures remain outside the repository/wheel; no models or recordings
are bundled or implicitly fetched by these workflows.
