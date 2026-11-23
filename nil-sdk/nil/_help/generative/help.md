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
