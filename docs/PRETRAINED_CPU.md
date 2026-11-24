# Pinned pretrained CPU acceptance

This gate uses the actual pretrained
[HuggingFaceTB/SmolLM2-135M model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
revision93efa2f097d58c2a74874c7e644dbc9b0cee75a2, with installed Transformers/
PEFT/Torch CPU backends. The checkpoint identifies LlamaForCausalLM and contains
134,515,008 unique parameters. Its model card declares Apache2.0. This is an
external pretrained checkpoint, separate from the earlier tiny from-scratch tests.

The explicit validation download was271,971,074 bytes, including safetensors and
JSON tokenizer/config files. Files stay under workspace artifacts/nil/models,
outside the SDK/repository and distribution. The acceptance stages verify pinned file hashes and never download anything;
only the explicit preparation mode can contact the model host.
Installing/importing NIL still installs only nil-sdk and does not load a model.

## Actual measured result — 2026-11-23

All stages ran in fresh subprocesses in one fresh transient Linux user unit:
CPUQuota100%, MemoryMax2GiB, MemorySwapMax0, TasksMax128, one backend thread.
All pass. Kernel unit peak was1,705,091,072 bytes (about1.59GiB), with zero OOM
kills and zero swap. Installed-wheel unit wall time44.61s includes startup and file checks.

| Stage | Observed result | Child peak RSS |
| --- | --- | --- |
| FP32 inference | Finite token-weighted NLL4.416751; deterministic8-token generation repeated exactly | 1138.66MiB |
| LoRA | 230,400 trainable parameters; eight updates; frozen base unchanged; adapter-only local reload reproduced NLL4.119316 | 1392.58MiB |
| INT8 weight storage | NLL4.564647, delta+0.147897 within the predefined+0.5 limit; state file220,317,635 bytes | 1438.43MiB |

The FP32 model parameter storage is538,060,032 bytes. Weight-only state stores
INT8 Linear weights plus scales, while embeddings/output head stay floating-point.
This is storage reduction, not proof of lower latency or an INT8-compute kernel.
Process RSS and cgroup charges differ because shared resident mappings and file
cache are accounted differently. Raw metrics and hashes are in
[the full report](benchmarks/2026-11-23-pretrained-135m.json).

The data consists of eight short fixed English windows (79 evaluated next-token
targets after tokenization/masking). LoRA adapts and evaluates those same windows:
this proves parameter updates, frozen-base ownership and reload fidelity, not
held-out language-quality improvement. It uses CPU FP32, eager attention, context32,
LoRA rank4/alpha8 targeting q_proj/v_proj and one epoch with batch1. The adapter
artifact includes tokenizer/config data and is about5.74MB. Temporary full training
checkpoints are removed by the parent-owned scratch directory even after a worker
failure or deadline. No permanent service was installed.

## Quantization acceptance and failures

Dynamic activation INT8 did not meet this checkpoint's +0.5 NLL tolerance. The
per-channel/head-excluded variant measured NLL6.810672 (+2.393922), so it remains
a failed configuration in the report. The threshold was not relaxed. The existing
small digit/MLP dynamic INT8 checks do not imply transformer fidelity.

`nil.optimization.quantize(..., method='weight_only_int8', exclude=('lm_head',))`
adds explicit per-channel INT8 weight storage with FP32 dequantized Linear math.
It preserves caller ownership and supports ordinary state-dict restoration into
the same converted architecture. There is no persistent dequantized weight cache.
Default dynamic_int8 remains available; it now supports explicit subtree exclusion
and Linear per-channel settings. Unknown paths/configurations fail before copying.
The helper converts one owned clone in place, removing the previous second full
model clone; caller weights/mode remain unchanged.

## Reproduction and loading contract

Prepare explicitly with the bounded public-checkpoint helper (this is the only
step permitted to download; it verifies sizes/revision and safetensors hashes):

```sh
python scripts/accept_pretrained.py \
  --prepare-model HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --snapshot /absolute/pinned-snapshot --max-download-mib 350
```

The helper records nil-source.json, refuses another revision in an occupied
directory, validates same-revision cached files without network access and
requires a single-file safetensors checkpoint within the explicit byte budget.
No checkpoint weights/tokenizer data are bundled with NIL. Once prepared, run:

```sh
systemd-run --user --wait --pipe --collect \
  --unit=nil-pretrained-cpu-check \
  -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=128 \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false \
  --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
  /absolute/venv/bin/python /absolute/NIL/scripts/accept_pretrained.py \
  --snapshot /absolute/pinned-snapshot --require-unit-policy \
  --output /absolute/results/pretrained.json
```

Use a regular NIL wheel and explicit generative backend. The runner supports
--stages inference/lora/int8 independently and refuses a target below100M
parameters. It records actual package versions, source hashes, logs, stage exit
codes, numerical outcomes and kernel policy/usage. Missing hardware/backend,
policy, snapshot integrity or numerical fidelity cannot produce acceptance.

For an explicit remote load, load_causal_lm(..., allow_download=True,
revision=commit_sha, cache_dir=chosen_directory) sends the same pinned revision,
cache and optional token to model and tokenizer; previously those options reached
only the model. Default loading remains local-only, allow_download must be a
boolean, and remote model code stays disabled. No global cache/driver is modified.

This closes the declared135M CPU pretrained workflow. Billion-parameter/GPU
models, other architectures, long contexts, cached-generation export, broad
language quality, public deployment and hard GPU VRAM quotas remain separate
hardware/model/deployment targets. No global percentage of AI coverage is implied.
