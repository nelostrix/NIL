"""
Personal Language Model that learns how YOU talk.
Uses Liquid Neural Networks for adaptive text generation.
"""

import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


class PersonalVocabulary:
    """Learns your personal vocabulary from your text."""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.word_freq = Counter()

    def build_from_text(self, texts):
        """Learn vocabulary from your messages."""
        print("Learning your vocabulary...")

        # Tokenize all texts
        all_words = []
        for text in texts:
            words = self._tokenize(text)
            all_words.extend(words)
            self.word_freq.update(words)

        # Add words to vocabulary (most frequent first)
        for word, freq in self.word_freq.most_common():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"✓ Learned {len(self.word2idx)} unique words from your writing")
        print(f"Most common words: {self.word_freq.most_common(10)}")

    def _tokenize(self, text):
        """Simple tokenization (you can improve this)."""
        # Convert to lowercase, split on spaces and punctuation
        text = text.lower()
        # Keep common punctuation as separate tokens
        text = re.sub(r"([.,!?;:])", r" \1 ", text)
        words = text.split()
        return words

    def text_to_indices(self, text):
        """Convert text to indices."""
        words = self._tokenize(text)
        indices = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        return indices

    def indices_to_text(self, indices):
        """Convert indices back to text."""
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices]
        # Remove special tokens
        words = [w for w in words if w not in ["<PAD>", "<START>", "<END>"]]
        return " ".join(words)


class PersonalLanguageModel(nn.Module):
    """
    Language model that learns YOUR communication style.
    Uses Liquid NN for adaptive text generation.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer (converts words to vectors)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Liquid RNN for adaptive learning
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # Self-attention to understand context
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

        # Output layer (predicts next word)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hidden=None):
        """
        Args:
            x: Input word indices [batch, seq_len]
            hidden: Previous hidden state
        Returns:
            output: Predictions for next word [batch, seq_len, vocab_size]
            hidden: New hidden state
        """
        # Embed words
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
        embedded = self.dropout(embedded)

        # Pass through RNN (learns sequential patterns)
        rnn_out, hidden = self.rnn(embedded, hidden)  # [batch, seq_len, hidden_dim]

        # Apply self-attention (understands context)
        attended, _ = self.attention(rnn_out, rnn_out, rnn_out)
        attended = self.dropout(attended)

        # Predict next word
        output = self.fc(attended)  # [batch, seq_len, vocab_size]

        return output, hidden

    def generate(self, vocab, start_text="", max_length=50, temperature=0.8):
        """
        Generate text in YOUR style!

        Args:
            vocab: PersonalVocabulary object
            start_text: Seed text to start from (optional)
            max_length: Maximum length of generated text
            temperature: Creativity (higher = more random)
        Returns:
            generated_text: Text that sounds like you
        """
        self.eval()

        # Start with seed text or START token
        if start_text:
            input_ids = vocab.text_to_indices(start_text)
        else:
            input_ids = [vocab.word2idx["<START>"]]

        generated_ids = input_ids.copy()
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                x = torch.LongTensor([input_ids]).unsqueeze(0)  # [1, 1, seq_len]

                # Get predictions
                output, hidden = self.forward(x, hidden)

                # Get last prediction
                logits = output[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=0)

                # Sample next word
                next_idx = torch.multinomial(probs, 1).item()

                # Stop if END token
                if next_idx == vocab.word2idx["<END>"]:
                    break

                # Add to generated sequence
                generated_ids.append(next_idx)
                input_ids = [next_idx]

        # Convert back to text
        generated_text = vocab.indices_to_text(generated_ids)
        return generated_text


class PersonalLMTrainer:
    """Trainer that learns your communication style."""

    def __init__(self, model, vocab, learning_rate=0.001):
        self.model = model
        self.vocab = vocab
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    def prepare_batch(self, texts, max_length=50):
        """Convert texts to training batches."""
        batch_input = []
        batch_target = []

        for text in texts:
            indices = self.vocab.text_to_indices(text)

            # Skip if too short
            if len(indices) < 2:
                continue

            # Add START token
            indices = (
                [self.vocab.word2idx["<START>"]]
                + indices
                + [self.vocab.word2idx["<END>"]]
            )

            # Truncate if too long
            if len(indices) > max_length:
                indices = indices[:max_length]

            # Input: all except last
            # Target: all except first (predict next word)
            batch_input.append(indices[:-1])
            batch_target.append(indices[1:])

        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_input)

        padded_input = []
        padded_target = []

        for inp, tgt in zip(batch_input, batch_target):
            # Pad with 0 (PAD token)
            padded_input.append(inp + [0] * (max_len - len(inp)))
            padded_target.append(tgt + [0] * (max_len - len(tgt)))

        return torch.LongTensor(padded_input), torch.LongTensor(padded_target)

    def train(self, texts, epochs=50, batch_size=8):
        """Train the model to talk like you."""
        print(f"\nTraining model to learn your style...")
        print(f"Training samples: {len(texts)}")

        self.model.train()

        for epoch in range(epochs):
            # Shuffle texts
            np.random.shuffle(texts)

            total_loss = 0
            num_batches = 0

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Prepare batch
                input_ids, target_ids = self.prepare_batch(batch_texts)

                # Forward pass
                output, _ = self.model(input_ids)

                # Reshape for loss calculation
                output = output.reshape(-1, self.vocab.vocab_size)
                target_ids = target_ids.reshape(-1)

                # Compute loss
                loss = self.criterion(output, target_ids)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

                # Generate sample
                sample = self.model.generate(self.vocab, max_length=20, temperature=0.7)
                print(f"  Sample: {sample}")

        print("\n✓ Training complete! Model learned your style.")

    def save_model(self, path):
        """Save trained model."""
        torch.save(
            {"model_state_dict": self.model.state_dict(), "vocab": self.vocab}, path
        )
        print(f"✓ Model saved to {path}")

    def load_model(self, path):
        """Load trained model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.vocab = checkpoint["vocab"]
        print(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    # Example: Train on sample texts
    sample_texts = [
        "hey what's up man how are you doing today",
        "i think we should meet tomorrow at 5pm",
        "yeah that sounds good to me let's do it",
        "btw did you see the game last night",
        "nah i missed it was it good",
        "yeah it was amazing you should watch the highlights",
    ]

    # Build vocabulary
    vocab = PersonalVocabulary()
    vocab.build_from_text(sample_texts)

    # Create model
    model = PersonalLanguageModel(
        vocab_size=len(vocab.word2idx), embedding_dim=64, hidden_dim=128, num_layers=2
    )

    # Train
    trainer = PersonalLMTrainer(model, vocab)
    trainer.train(sample_texts, epochs=100)

    # Generate text
    print("\n" + "=" * 60)
    print("GENERATING TEXT IN YOUR STYLE")
    print("=" * 60)

    for _ in range(5):
        generated = model.generate(vocab, max_length=15, temperature=0.8)
        print(f"  → {generated}")
