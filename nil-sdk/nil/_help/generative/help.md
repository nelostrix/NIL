# generative

Build/load local causal language models, prepare text, apply optional LoRA, and measure perplexity.

## Usage

```python
import nil

tokenizer = nil.generative.tokenizer_from_texts(["a small local example"] * 4, vocab_size=300)
model = nil.generative.causal_lm(len(tokenizer), context_length=32, width=32, layers=1, heads=2)
print(type(model).__name__)
```

## Parameters

Tokenizer vocabulary, model dimensions and local paths. LoRA accepts rank and target modules.

## Output

Native Transformers/PEFT models and tokenizers.

## Requirements

nil-sdk[generative].

## Limitations

No automatic pretrained download. Tiny examples produce no useful language quality guarantee. Distributed/RLHF/QLoRA training is not implemented.

## Pinned pretrained loading

`load_causal_lm(path_or_id, revision=commit_sha, cache_dir=explicit_directory,
allow_download=False, **model_options)` passes the same revision, cache and
optional token to both model and tokenizer. Downloads remain opt-in and remote
model code stays disabled. For repeatable remote loads, select an immutable
revision and request safetensors explicitly. Local snapshots work offline.

A real134.5M-parameter SmolLM2/Llama checkpoint passed FP32 inference, deterministic
generation, LoRA on q_proj/v_proj, frozen-base checks and adapter-only reload
under one-CPU/2GiB/no-swap limits. This is declared model-specific CPU evidence;
large/GPU/arbitrary architectures remain separate targets. See
 docs/PRETRAINED_CPU.md and the raw pinned report in docs/benchmarks.
