# ============================================================
#  MindPulse — Text Preprocessing Pipeline
#  Author  : Hanzla (NLP Specialist & QA Lead)
#  File    : data/text_preprocessing.py
#  Purpose : Clean and prepare raw Reddit text (SMHD dataset)
#            for LSTM and BERT model training
# ============================================================

import re
import string
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Optional


# ── Configuration ────────────────────────────────────────────
MAX_SEQUENCE_LENGTH = 256    # Max tokens per post for LSTM
MAX_VOCABULARY_SIZE = 20000  # Top N most frequent words
PADDING_TOKEN       = "<PAD>"
UNKNOWN_TOKEN       = "<UNK>"
MIN_POST_LENGTH     = 20     # Minimum words — shorter posts are dropped

# Label mapping
LABEL_MAP = {
    "control"    : 0,
    "depression" : 1,
    "anxiety"    : 2
}


class TextPreprocessor:
    """
    Cleans raw Reddit text and converts it into integer
    sequences ready for the LSTM model.

    Usage:
        tp = TextPreprocessor()
        tp.build_vocab(list_of_clean_texts)
        encoded = tp.process("I feel really anxious today")
    """

    def __init__(self):
        self.word_to_index : Dict[str, int] = {}
        self.index_to_word : Dict[int, str] = {}
        self.vocab_size    : int = 0

    def clean_text(self, text: str) -> str:
        """
        Remove noise from a raw Reddit post.

        Removes:
          - URLs (http, https, www)
          - Reddit @mentions and r/subreddit links
          - HTML entities (&amp; etc.)
          - Punctuation (keeps apostrophes for contractions)
          - Extra whitespace

        Args:
            text: Raw Reddit post string

        Returns:
            Cleaned lowercase string, or "" if input is invalid
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)        # remove URLs
        text = re.sub(r"@\w+|r/\w+|u/\w+", "", text)       # remove mentions
        text = re.sub(r"&\w+;", "", text)                    # remove HTML entities

        # Keep letters, digits, spaces, and apostrophes only
        keep = set(string.ascii_lowercase + string.digits + " '")
        text = "".join(ch if ch in keep else " " for ch in text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def is_valid_post(self, text: str) -> bool:
        """
        Returns True if the post has at least MIN_POST_LENGTH words.
        Very short posts don't contain enough signal for the model.
        """
        return len(text.split()) >= MIN_POST_LENGTH

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build a word-to-integer dictionary from a list of clean texts.

        Index 0 = <PAD> (padding)
        Index 1 = <UNK> (unknown words)
        Index 2+ = actual words, most frequent first

        Args:
            texts: List of already-cleaned text strings
        """
        print(f"Building vocabulary from {len(texts):,} posts...")

        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        # Keep top N words (minus 2 reserved spots)
        most_common = word_counts.most_common(MAX_VOCABULARY_SIZE - 2)

        self.word_to_index = {PADDING_TOKEN: 0, UNKNOWN_TOKEN: 1}
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_index[word] = idx

        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.vocab_size = len(self.word_to_index)

        print(f"Vocabulary ready: {self.vocab_size:,} tokens")

    def encode(self, text: str) -> List[int]:
        """
        Convert a clean text string to a list of integers.

        Example:
            "i feel anxious" → [45, 12, 389]

        Unknown words map to index 1 (<UNK>).
        """
        if not self.word_to_index:
            raise RuntimeError("Call build_vocab() before encode()")

        return [self.word_to_index.get(w, 1) for w in text.split()]

    def pad_sequence(self, sequence: List[int]) -> np.ndarray:
        """
        Make all sequences exactly MAX_SEQUENCE_LENGTH tokens long.

        Short sequences → pad with zeros at the end
        Long sequences  → keep the LAST N tokens
                          (Reddit posts usually end with the most
                           emotionally relevant content)

        Returns:
            numpy array of shape (MAX_SEQUENCE_LENGTH,)
        """
        if len(sequence) >= MAX_SEQUENCE_LENGTH:
            return np.array(sequence[-MAX_SEQUENCE_LENGTH:])
        else:
            padding = MAX_SEQUENCE_LENGTH - len(sequence)
            return np.array(sequence + [0] * padding)

    def process(self, text: str) -> Optional[np.ndarray]:
        """
        Full pipeline: raw text → padded integer array.

        Returns None if the post is too short to be useful.
        """
        clean = self.clean_text(text)
        if not self.is_valid_post(clean):
            return None
        return self.pad_sequence(self.encode(clean))


def prepare_smhd_dataset(
    raw_texts : List[str],
    labels    : List[int],
) -> Tuple[np.ndarray, np.ndarray, TextPreprocessor]:
    """
    Full pipeline for the SMHD Reddit dataset.

    Steps:
        1. Clean all texts
        2. Filter posts that are too short
        3. Build vocabulary
        4. Encode and pad all texts

    Args:
        raw_texts : List of raw Reddit post strings
        labels    : List of integer labels
                    (0=control, 1=depression, 2=anxiety)

    Returns:
        X         : numpy array shape (N, MAX_SEQUENCE_LENGTH)
        y         : numpy array shape (N,)
        processor : fitted TextPreprocessor — SAVE THIS for inference

    Example:
        X, y, processor = prepare_smhd_dataset(posts, labels)
        # X.shape → (num_posts, 256)
    """
    processor = TextPreprocessor()

    print("Step 1/4 — Cleaning texts...")
    cleaned = [processor.clean_text(t) for t in raw_texts]

    print("Step 2/4 — Filtering short posts...")
    valid_pairs = [
        (text, label)
        for text, label in zip(cleaned, labels)
        if processor.is_valid_post(text)
    ]
    if not valid_pairs:
        raise ValueError("No valid posts found after filtering!")

    valid_texts, valid_labels = zip(*valid_pairs)
    print(f"  Kept {len(valid_texts):,} of {len(raw_texts):,} posts")

    print("Step 3/4 — Building vocabulary...")
    processor.build_vocab(list(valid_texts))

    print("Step 4/4 — Encoding and padding...")
    X = np.array([
        processor.pad_sequence(processor.encode(t))
        for t in valid_texts
    ])
    y = np.array(valid_labels)

    print(f"\nDone — X shape: {X.shape}, y shape: {y.shape}")
    return X, y, processor


# ── Quick test — run this file directly to check it works ────
if __name__ == "__main__":
    print("=" * 50)
    print("Testing TextPreprocessor")
    print("=" * 50)

    sample_posts = [
        "I have been feeling really hopeless and sad and cannot get out of bed every single day for many weeks now",
        "Everything makes me so anxious all the time my heart races constantly and i cannot sleep properly at night",
        "Had a great day today went to the park with my friends and felt really happy and relaxed for once",
        "Too short",   # This will be filtered out
        "Visit https://example.com and check the resource posted by user on the mental health forum today please",
    ]
    sample_labels = [1, 2, 0, 0, 0]

    X, y, processor = prepare_smhd_dataset(sample_posts, sample_labels)

    print(f"\nFirst post encoded (first 10 tokens): {X[0][:10]}")
    print(f"Label: {y[0]} (1=depression)")
    print(f"Vocab size: {processor.vocab_size}")
    print("\nAll tests passed!")
