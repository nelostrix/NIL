"""
Online Learning Language Model.
Learns new words and patterns in real-time during conversation.
Supports Hinglish (Hindi + English mix).
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import re
import json
from pathlib import Path


class AdaptiveVocabulary:
    """
    Vocabulary that grows as you talk.
    Learns new words on-the-fly!
    """

    def __init__(self):
        # Core tokens
        self.word2idx = {
            '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3
        }
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}

        # Track word usage frequency
        self.word_freq = defaultdict(int)

        # Track new words added during conversation
        self.new_words_this_session = []

        # Common Hinglish patterns
        self.hinglish_patterns = {
            'yaar': 'friend',
            'bhai': 'brother',
            'kya': 'what',
            'hai': 'is',
            'nahi': 'no',
            'haan': 'yes',
            'acha': 'okay',
            'theek': 'fine',
            'kaise': 'how',
            'kaha': 'where',
            'kyun': 'why',
            'matlab': 'means',
            'bas': 'just',
            'abhi': 'now',
            'kar': 'do',
            'ho': 'are',
            'na': 'no/right',
            'chal': 'okay/come',
            'kal': 'tomorrow/yesterday',
            'aaj': 'today'
        }

    def add_word(self, word):
        """Add a new word to vocabulary (real-time learning!)."""
        if word not in self.word2idx and word.strip():
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.new_words_this_session.append(word)
            print(f"  [LEARNED NEW WORD] '{word}' (vocab size: {len(self.word2idx)})")
            return True
        return False

    def tokenize(self, text):
        """
        Tokenize with Hinglish support.
        Handles English, Hindi (romanized), and mixed.
        """
        # Lowercase
        text = text.lower()

        # Handle common contractions
        text = text.replace("'s", " is")
        text = text.replace("n't", " not")
        text = text.replace("'re", " are")
        text = text.replace("'ve", " have")
        text = text.replace("'ll", " will")

        # Keep punctuation as separate tokens
        text = re.sub(r"([.,!?;:])", r" \1 ", text)

        # Split on whitespace
        tokens = text.split()

        # Filter empty tokens
        tokens = [t for t in tokens if t.strip()]

        return tokens

    def text_to_indices(self, text, learn_new_words=False):
        """
        Convert text to indices.
        If learn_new_words=True, adds unknown words to vocabulary.
        """
        tokens = self.tokenize(text)
        indices = []

        for token in tokens:
            if token in self.word2idx:
                # Known word
                indices.append(self.word2idx[token])
                self.word_freq[token] += 1
            else:
                # Unknown word
                if learn_new_words:
                    # Learn it!
                    self.add_word(token)
                    indices.append(self.word2idx[token])
                    self.word_freq[token] = 1
                else:
                    # Use UNK token
                    indices.append(self.word2idx['<UNK>'])

        return indices

    def indices_to_text(self, indices):
        """Convert indices back to text."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<START>', '<END>']:
                words.append(word)

        # Join and clean up spacing around punctuation
        text = ' '.join(words)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        return text

    def save(self, path):
        """Save vocabulary state."""
        data = {
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'word_freq': dict(self.word_freq)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Vocabulary saved ({len(self.word2idx)} words)")

    def load(self, path):
        """Load vocabulary state."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self.word_freq = defaultdict(int, data['word_freq'])
        print(f"✓ Vocabulary loaded ({len(self.word2idx)} words)")


class OnlineLearningLM(nn.Module):
    """
    Language
