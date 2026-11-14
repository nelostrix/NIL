"""Causal model development: local loading, text preparation and optional LoRA.

No model downloads by default. Returned objects are native Transformers models
and tokenizers; arbitrary architectures can use Trainer(custom_step).
"""

from pathlib import Path

from nil.generative.evaluation import perplexity as perplexity
from nil.runtime import require


def tokenizer_from_texts(texts, vocab_size=1000):
    """Train a local byte-level BPE tokenizer; no remote vocabulary needed."""
    transformers = require("transformers", "generative")
    tokenizers = require("tokenizers", "generative")
    if vocab_size < 260:
        raise ValueError("Byte-level BPE vocabulary must have at least 260 tokens")
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(texts, trainer)
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


def causal_lm(vocab_size, context_length=256, width=128, layers=2, heads=4):
    """Construct a small GPT-2-style model from scratch; native HF return type."""
    transformers = require("transformers", "generative")
    if min(vocab_size, context_length, width, layers, heads) < 1 or width % heads:
        raise ValueError("Positive dimensions required; width must be divisible by heads")
    config = transformers.GPT2Config(
        vocab_size=vocab_size,
        n_positions=context_length,
        n_ctx=context_length,
        n_embd=width,
        n_layer=layers,
        n_head=heads,
        bos_token_id=None,
        eos_token_id=None,
    )
    return transformers.AutoModelForCausalLM.from_config(config)


def load_causal_lm(path, *, allow_download=False, **model_kwargs):
    transformers = require("transformers", "generative")
    if model_kwargs.pop("trust_remote_code", False):
        raise ValueError("Automatic execution of remote model code is not supported")
    if "local_files_only" in model_kwargs:
        raise ValueError("Use allow_download to control network access")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=not allow_download, trust_remote_code=False, **model_kwargs
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(path), local_files_only=not allow_download, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def text_dataset(texts, tokenizer, max_length=256):
    """Fixed-length causal batches; attention masks distinguish EOS from padding."""
    torch = require("torch", "torch")
    from nil.data import Records

    if max_length < 2:
        raise ValueError("max_length must be at least two")
    records = []
    for text in texts:
        encoded = tokenizer(
            text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        ids = encoded["input_ids"][0]
        mask = encoded["attention_mask"][0]
        if int(mask.sum()) < 2:
            raise ValueError("Each causal training example requires at least two tokens")
        labels = ids.clone()
        labels[mask == 0] = -100
        # The model shifts internally. Position zero is never a prediction target.
        if not torch.any(labels[1:] != -100).item():
            raise ValueError("Example has no causal prediction targets")
        records.append({"input_ids": ids, "attention_mask": mask, "labels": labels})
    if not records:
        raise ValueError("Provide at least one text example")
    return Records(records)


def lora(model, rank=8, alpha=16, dropout=0.0, target_modules=None):
    peft = require("peft", "generative")
    if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("LoRA requires rank>0, alpha>0 and dropout in [0,1)")
    config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return peft.get_peft_model(model, config)


def generate(model, tokenizer, prompt, max_new_tokens=64, **kwargs):
    torch = require("torch", "torch")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    inputs = tokenizer(prompt, return_tensors="pt")
    target = next(model.parameters()).device
    inputs = {key: value.to(target) for key, value in inputs.items()}
    previous = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    finally:
        model.train(previous)


def save(model, tokenizer, path):
    """Save native Transformers/PEFT artifacts (safetensors where supported)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))
