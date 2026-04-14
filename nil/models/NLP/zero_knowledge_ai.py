"""
Zero-Knowledge Conversational AI.
Starts knowing NOTHING. Learns EVERYTHING from your voice.
Like a human child learning language.
"""

import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BabyAI:
    """
    AI that starts with zero knowledge.
    Learns language purely from your conversations.
    """

    def __init__(self, memory_size=10000):
        """
        Initialize with blank slate - knows NOTHING.
        """
        # === KNOWLEDGE BASE (empty at start) ===
        self.vocabulary = {}  # word -> unique_id
        self.word_id_counter = 0

        # === MEMORY (stores what you said) ===
        self.conversation_history = deque(maxlen=memory_size)
        self.word_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.word_frequency = defaultdict(int)

        # === PATTERN RECOGNITION ===
        self.phrase_patterns = defaultdict(int)  # "how are" -> count
        self.response_patterns = {}  # "how are you" -> ["i'm good", "fine"]

        # === CONTEXT UNDERSTANDING ===
        self.context_memory = []  # Last N things you said
        self.context_window = 5

        # === LEARNING STATS ===
        self.total_conversations = 0
        self.words_learned = 0
        self.patterns_discovered = 0

        print("🍼 Baby AI initialized - knows NOTHING")
        print("Ready to learn from your voice!\n")

    def hear(self, text):
        """
        You speak -> AI hears and learns.
        This is the ONLY way it gets data!
        """
        print(f"\n👂 Heard you say: '{text}'")

        # Break into words
        words = self._tokenize(text)

        # Learn each word
        new_words = []
        for word in words:
            if self._learn_word(word):
                new_words.append(word)

        if new_words:
            print(f"  📚 Learned new words: {new_words}")
            print(f"  📊 Total vocabulary: {len(self.vocabulary)} words")

        # Learn word patterns (which words appear together)
        self._learn_patterns(words)

        # Store in conversation history
        self.conversation_history.append(
            {"timestamp": datetime.now().isoformat(), "text": text, "words": words}
        )

        # Update context memory
        self.context_memory.append(words)
        if len(self.context_memory) > self.context_window:
            self.context_memory.pop(0)

        self.total_conversations += 1

    def _tokenize(self, text):
        """
        Break text into words.
        Handles English + Hindi romanized (Hinglish).
        """
        # Lowercase
        text = text.lower()

        # Simple split on spaces and punctuation
        import re

        # Keep punctuation as part of words if needed
        words = re.findall(r"\b\w+\b", text)

        return words

    def _learn_word(self, word):
        """
        Learn a new word (if not known).
        Returns True if word was new.
        """
        if word not in self.vocabulary:
            self.vocabulary[word] = self.word_id_counter
            self.word_id_counter += 1
            self.words_learned += 1
            return True

        # Already know this word, just count it
        self.word_frequency[word] += 1
        return False

    def _learn_patterns(self, words):
        """
        Discover which words appear together (co-occurrence).
        This is how it learns grammar/structure!
        """
        # Bigrams (2-word patterns)
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            pattern = f"{word1} {word2}"
            self.phrase_patterns[pattern] += 1
            self.word_cooccurrence[word1][word2] += 1

        # Trigrams (3-word patterns)
        for i in range(len(words) - 2):
            pattern = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            self.phrase_patterns[pattern] += 1

        self.patterns_discovered = len(self.phrase_patterns)

    def think(self):
        """
        AI generates a response based on what it learned.
        Uses ONLY words and patterns from YOUR speech.
        """
        if len(self.vocabulary) < 3:
            return "..."  # Not enough words learned yet

        # Strategy: Use most common patterns you use
        response_words = []

        # Start with a common word from recent context
        if self.context_memory:
            recent_words = [w for conv in self.context_memory for w in conv]
            if recent_words:
                # Pick a word you use often
                word_counts = defaultdict(int)
                for w in recent_words:
                    word_counts[w] += 1

                start_word = max(word_counts.items(), key=lambda x: x[1])[0]
                response_words.append(start_word)

        # Continue with words that usually follow
        max_length = 8
        while len(response_words) < max_length:
            if not response_words:
                break

            current_word = response_words[-1]

            # Find words that usually come after current_word
            next_words = self.word_cooccurrence[current_word]

            if not next_words:
                break

            # Pick most common next word
            next_word = max(next_words.items(), key=lambda x: x[1])[0]
            response_words.append(next_word)

            # Stop at natural end (punctuation or short response)
            if next_word in [".", "?", "!"] or len(response_words) >= 5:
                break

        response = " ".join(response_words)
        return response if response else "..."

    def report_learning(self):
        """Show what AI has learned so far."""
        print("\n" + "=" * 60)
        print("🧠 LEARNING REPORT")
        print("=" * 60)
        print(f"Total conversations: {self.total_conversations}")
        print(f"Words learned: {self.words_learned}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Patterns discovered: {self.patterns_discovered}")

        if self.word_frequency:
            print(f"\nMost used words:")
            top_words = sorted(
                self.word_frequency.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for word, count in top_words:
                print(f"  - '{word}': {count} times")

        if self.phrase_patterns:
            print(f"\nCommon phrases you use:")
            top_patterns = sorted(
                self.phrase_patterns.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for pattern, count in top_patterns:
                print(f"  - '{pattern}': {count} times")

        print("=" * 60 + "\n")

    def save_knowledge(self, path):
        """Save everything learned to disk."""
        knowledge = {
            "vocabulary": self.vocabulary,
            "word_frequency": dict(self.word_frequency),
            "phrase_patterns": dict(self.phrase_patterns),
            "word_cooccurrence": {
                k: dict(v) for k, v in self.word_cooccurrence.items()
            },
            "conversation_history": list(self.conversation_history),
            "stats": {
                "total_conversations": self.total_conversations,
                "words_learned": self.words_learned,
                "patterns_discovered": self.patterns_discovered,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        print(f"✓ Knowledge saved to {path}")

    def load_knowledge(self, path):
        """Load previously learned knowledge."""
        with open(path, "r", encoding="utf-8") as f:
            knowledge = json.load(f)

        self.vocabulary = knowledge["vocabulary"]
        self.word_frequency = defaultdict(int, knowledge["word_frequency"])
        self.phrase_patterns = defaultdict(int, knowledge["phrase_patterns"])
        self.word_cooccurrence = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in knowledge["word_cooccurrence"].items()},
        )
        self.conversation_history = deque(
            knowledge["conversation_history"], maxlen=10000
        )

        stats = knowledge["stats"]
        self.total_conversations = stats["total_conversations"]
        self.words_learned = stats["words_learned"]
        self.patterns_discovered = stats["patterns_discovered"]

        print(f"✓ Knowledge loaded from {path}")
        print(f"  - {len(self.vocabulary)} words known")
        print(f"  - {self.total_conversations} conversations remembered")


if __name__ == "__main__":
    # Create blank AI
    baby = BabyAI()

    # Simulate learning from conversations
    print("=== LEARNING SESSION ===\n")

    conversations = [
        "hey what's up",
        "how are you doing today",
        "i'm doing good thanks",
        "what are you working on",
        "just working on some code",
        "nice what kind of code",
        "building an AI that learns from scratch",
        "that sounds cool",
        "yeah it's pretty interesting",
        "how does it work",
        "it learns from conversations like this one",
        "wow that's amazing",
    ]

    for conv in conversations:
        baby.hear(conv)

        # After a few conversations, try responding
        if len(baby.vocabulary) > 5:
            response = baby.think()
            print(f"  🤖 AI thinks: '{response}'")

    # Show what it learned
    baby.report_learning()

    # Save knowledge
    baby.save_knowledge("knowledge/baby_ai_session1.json")
