"""Build and fine-tune a tiny causal LM entirely locally. No pretrained weights."""

import argparse
import json

import torch

import nil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./nil_runs")
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()
    nil.runtime.seed(42)
    torch.set_num_threads(1)
    corpus = [
        "the robot observes the room",
        "the robot stops before an obstacle",
        "the robot follows a safe path",
        "the operator approves the action",
    ] * 8
    tokenizer = nil.generative.tokenizer_from_texts(corpus, vocab_size=300)
    data = nil.generative.text_dataset(corpus, tokenizer, max_length=24)
    model = nil.generative.causal_lm(len(tokenizer), context_length=48, width=32, layers=1, heads=2)
    if args.lora:
        model = nil.generative.lora(model, rank=4, target_modules=["c_attn"])
    result = nil.train(
        model,
        data,
        task="causal_lm",
        epochs=4,
        batch_size=8,
        learning_rate=0.003,
        device="cpu",
        output_dir=args.output,
    )
    if args.lora:
        model = model.merge_and_unload()
    destination = result.run.path / "language-model"
    nil.generative.save(model, tokenizer, destination)
    loaded, tokenizer = nil.generative.load_causal_lm(destination)
    evaluation = nil.generative.perplexity(loaded, data, device_name="cpu")
    text = nil.generative.generate(
        loaded,
        tokenizer,
        "the robot",
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(
        json.dumps(
            {
                "run": str(result.run.path),
                "history": result.history,
                "saved_model": str(destination),
                "sample": text,
                "evaluation": evaluation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
