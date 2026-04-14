"""
Train a language model on YOUR personal writing style.
"""

import os
from pathlib import Path

from nil.models.personal_lm import (
    PersonalLanguageModel,
    PersonalLMTrainer,
    PersonalVocabulary,
)


def load_your_messages(file_path):
    """Load your messages from a text file."""
    print(f"Loading your messages from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"✓ Loaded {len(texts)} messages")
    return texts


def train_from_file(messages_file, model_save_path="models/my_language_model.pt"):
    """Train model on your messages."""

    # Load your messages
    texts = load_your_messages(messages_file)

    if len(texts) < 10:
        print("⚠️  Warning: Very few messages. Add more for better results!")

    # Build vocabulary from YOUR words
    vocab = PersonalVocabulary()
    vocab.build_from_text(texts)

    # Create model
    print("\nCreating language model...")
    model = PersonalLanguageModel(
        vocab_size=len(vocab.word2idx), embedding_dim=128, hidden_dim=256, num_layers=2
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = PersonalLMTrainer(model, vocab, learning_rate=0.001)
    trainer.train(texts, epochs=100, batch_size=16)

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trainer.save_model(model_save_path)

    return model, vocab


def interactive_generation(model, vocab):
    """Interactively generate text."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TEXT GENERATION")
    print("=" * 60)
    print("Commands:")
    print("  - Type text to use as seed")
    print("  - Type 'random' for random generation")
    print("  - Type 'quit' to exit")
    print("=" * 60)

    model.eval()

    while True:
        user_input = input("\n> ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "random":
            seed_text = ""
        else:
            seed_text = user_input

        # Generate
        generated = model.generate(
            vocab, start_text=seed_text, max_length=30, temperature=0.8
        )

        print(f"AI: {generated}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train personal language model")
    parser.add_argument(
        "--messages",
        type=str,
        required=True,
        help="Path to file with your messages (one per line)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/my_language_model.pt",
        help="Where to save trained model",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive generation after training",
    )

    args = parser.parse_args()

    # Train
    model, vocab = train_from_file(args.messages, args.model_path)

    # Interactive mode
    if args.interactive:
        interactive_generation(model, vocab)
    else:
        # Just show some samples
        print("\n" + "=" * 60)
        print("SAMPLE GENERATIONS")
        print("=" * 60)

        for i in range(10):
            generated = model.generate(vocab, max_length=20, temperature=0.8)
            print(f"{i + 1}. {generated}")


if __name__ == "__main__":
    main()
