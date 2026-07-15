"""
================================================================
  MindPulse — AI Mental Wellbeing System
  Complete Project — Single File Version
  
  Team:
    Sahar   — Group Leader
    Ahmad   — Deputy Leader & Database
    Puja    — ML Engineer (LSTM + Hybrid)
    Hanzla  — NLP Specialist & QA Lead (BERT)
    Usama   — App Developer (Streamlit)
    Irfan   — Ethics & Bias Lead

  Datasets:
    SMHD    — Reddit posts (depression/anxiety/control)
    DASS-21 — Questionnaire responses
    WESAD   — Physiological signals (ECG, EDA, temp)
    Custom  — Simulated dataset

  Models:
    1. LSTM  — Text classification
    2. BERT  — Fine-tuned NLP model
    3. Hybrid — NLP + physiological signal fusion

  Run app:
    streamlit run mindpulse_complete.py

  Run tests:
    pytest mindpulse_complete.py -v

  Ethics Notice:
    This system is a screening tool only.
    NOT a medical diagnosis.
    Always consult a qualified professional.
================================================================
"""

# ── Standard imports ──────────────────────────────────────────
import os
import re
import json
import uuid
import string
import hashlib
import secrets
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Dict, Optional

# ── Signal processing ─────────────────────────────────────────
from scipy import signal as scipy_signal

# ── Database ──────────────────────────────────────────────────
from sqlalchemy import (
    create_engine, Column, String, Integer,
    Float, DateTime, Boolean, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Sklearn ───────────────────────────────────────────────────
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

# ── Streamlit ─────────────────────────────────────────────────
import streamlit as st


# ================================================================
#  SECTION 1 — CONFIGURATION & CONSTANTS
# ================================================================

# Text preprocessing
MAX_SEQUENCE_LENGTH = 256
MAX_VOCABULARY_SIZE = 20000
PADDING_TOKEN       = "<PAD>"
UNKNOWN_TOKEN       = "<UNK>"
MIN_POST_LENGTH     = 20

# DASS-21
DEPRESSION_ITEMS = [3, 5, 10, 13, 16, 17, 21]
ANXIETY_ITEMS    = [2, 4,  7,  9, 15, 19, 20]
STRESS_ITEMS     = [1, 6,  8, 11, 12, 14, 18]
THRESHOLDS       = {"depression": 14, "anxiety": 10, "stress": 19}
MAX_DASS_SCORE   = 42

# WESAD physiological signals
WESAD_SAMPLING_RATES = {"ecg": 700, "eda": 4, "temp": 4}
WINDOW_SECONDS       = 60
OVERLAP_SECONDS      = 30

# Model configuration
VOCAB_SIZE      = MAX_VOCABULARY_SIZE + 2
EMBEDDING_DIM   = 128
NUM_CLASSES     = 3
BATCH_SIZE      = 64
LEARNING_RATE   = 0.001
MAX_EPOCHS      = 50
CLASS_NAMES     = ["Control", "Depression", "Anxiety"]

# Label mapping
LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "stress": 3}

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///mindpulse.db")

# UI colours
COLORS = {
    "depression": "#993C1D",
    "anxiety"   : "#534AB7",
    "stress"    : "#854F0B",
    "control"   : "#0F6E56",
}


# ================================================================
#  SECTION 2 — TEXT PREPROCESSING (Hanzla)
# ================================================================

class TextPreprocessor:
    """
    Cleans raw Reddit text (SMHD dataset) and converts
    it into integer sequences for LSTM and BERT models.

    Pipeline:
        1. clean_text()    — remove noise
        2. is_valid_post() — filter short posts
        3. build_vocab()   — build word to integer mapping
        4. encode()        — convert text to integers
        5. pad_sequence()  — make all sequences same length
        6. process()       — full pipeline in one call
    """

    def __init__(self):
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}
        self.vocab_size: int = 0

    def clean_text(self, text: str) -> str:
        """
        Remove noise from raw Reddit post.

        Removes: URLs, @mentions, r/subreddit links,
                 HTML entities, punctuation, extra spaces.

        Args:
            text: Raw Reddit post string

        Returns:
            Clean lowercase string, or "" if invalid input
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+|r/\w+|u/\w+", "", text)
        text = re.sub(r"&\w+;", "", text)

        keep = set(string.ascii_lowercase + string.digits + " '")
        text = "".join(ch if ch in keep else " " for ch in text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def is_valid_post(self, text: str) -> bool:
        """Return True if post has at least MIN_POST_LENGTH words."""
        return len(text.split()) >= MIN_POST_LENGTH

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build word to integer dictionary from training texts.

        Index 0 = <PAD> (padding token)
        Index 1 = <UNK> (unknown words)
        Index 2+ = actual words, most frequent first

        Args:
            texts: List of already-cleaned text strings
        """
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        most_common = word_counts.most_common(MAX_VOCABULARY_SIZE - 2)
        self.word_to_index = {PADDING_TOKEN: 0, UNKNOWN_TOKEN: 1}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_index[word] = idx

        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.vocab_size = len(self.word_to_index)

    def encode(self, text: str) -> List[int]:
        """
        Convert clean text to list of integers.
        Unknown words map to index 1 (UNK).
        """
        if not self.word_to_index:
            raise RuntimeError("Call build_vocab() before encode()")
        return [self.word_to_index.get(w, 1) for w in text.split()]

    def pad_sequence(self, sequence: List[int]) -> np.ndarray:
        """
        Make sequence exactly MAX_SEQUENCE_LENGTH tokens.

        Short sequences: pad with zeros at the end.
        Long sequences : keep the LAST N tokens
                         (Reddit posts end with most emotional content)
        """
        if len(sequence) >= MAX_SEQUENCE_LENGTH:
            return np.array(sequence[-MAX_SEQUENCE_LENGTH:])
        padding = MAX_SEQUENCE_LENGTH - len(sequence)
        return np.array(sequence + [0] * padding)

    def process(self, text: str) -> Optional[np.ndarray]:
        """Full pipeline: raw text → padded integer array."""
        clean = self.clean_text(text)
        if not self.is_valid_post(clean):
            return None
        return self.pad_sequence(self.encode(clean))


def prepare_smhd_dataset(
    raw_texts: List[str],
    labels   : List[int],
) -> Tuple[np.ndarray, np.ndarray, TextPreprocessor]:
    """
    Full pipeline for SMHD Reddit dataset.

    Steps:
        1. Clean all texts
        2. Filter posts shorter than MIN_POST_LENGTH
        3. Build vocabulary from training data
        4. Encode and pad all texts

    Args:
        raw_texts: List of raw Reddit post strings
        labels   : List of integer labels (0/1/2)

    Returns:
        X         : Shape (N, MAX_SEQUENCE_LENGTH)
        y         : Shape (N,)
        processor : Fitted TextPreprocessor — save for inference!
    """
    processor = TextPreprocessor()

    cleaned = [processor.clean_text(t) for t in raw_texts]

    valid_pairs = [
        (text, label)
        for text, label in zip(cleaned, labels)
        if processor.is_valid_post(text)
    ]

    if not valid_pairs:
        raise ValueError("No valid posts found after filtering!")

    valid_texts, valid_labels = zip(*valid_pairs)
    processor.build_vocab(list(valid_texts))

    X = np.array([
        processor.pad_sequence(processor.encode(t))
        for t in valid_texts
    ])
    y = np.array(valid_labels)

    return X, y, processor


# ================================================================
#  SECTION 3 — DASS-21 PROCESSING (Puja)
# ================================================================

def compute_subscale_scores(answers: Dict[str, int]) -> Dict[str, int]:
    """
    Compute DASS-21 subscale scores from raw answers.

    Each subscale score = sum of 7 items × 2
    (multiplying by 2 scales to match original 42-item DASS norms)

    Args:
        answers: {"q1": 0..3, "q2": 0..3, ..., "q21": 0..3}

    Returns:
        {"depression": int, "anxiety": int, "stress": int}
        All values range from 0 to 42.
    """
    return {
        "depression": sum(answers.get(f"q{i}", 0) for i in DEPRESSION_ITEMS) * 2,
        "anxiety"   : sum(answers.get(f"q{i}", 0) for i in ANXIETY_ITEMS)    * 2,
        "stress"    : sum(answers.get(f"q{i}", 0) for i in STRESS_ITEMS)     * 2,
    }


def classify_dass_label(scores: Dict[str, int]) -> Tuple[str, float]:
    """
    Apply official DASS-21 clinical thresholds to assign label.

    Clinical thresholds (moderate severity):
        Depression >= 14
        Anxiety    >= 10
        Stress     >= 19

    Args:
        scores: Output of compute_subscale_scores()

    Returns:
        (label, confidence) — label is most elevated subscale
    """
    elevated = {
        name: score
        for name, score in scores.items()
        if score >= THRESHOLDS.get(name, 99)
    }

    if not elevated:
        return "control", round(1.0 - max(scores.values()) / MAX_DASS_SCORE, 3)

    label      = max(elevated, key=elevated.get)
    confidence = min(scores[label] / MAX_DASS_SCORE, 1.0)
    return label, round(confidence, 3)


def normalise_dass_scores(scores: Dict[str, int]) -> Dict[str, float]:
    """Normalise subscale scores to 0.0–1.0 range for neural network input."""
    return {k: round(v / MAX_DASS_SCORE, 4) for k, v in scores.items()}


def build_dass_feature_vector(answers: Dict[str, int]) -> np.ndarray:
    """
    Convert DASS-21 answers into a 24-feature vector.

    Features:
        21 raw item scores (one per question)
         3 normalised subscale scores
        Total = 24 features

    Returns:
        numpy array shape (24,)
    """
    raw_items = np.array([float(answers.get(f"q{i}", 0)) for i in range(1, 22)])
    scores    = compute_subscale_scores(answers)
    norm      = normalise_dass_scores(scores)
    subscales = np.array([norm["depression"], norm["anxiety"], norm["stress"]])
    return np.concatenate([raw_items, subscales])


# ================================================================
#  SECTION 4 — PHYSIOLOGICAL SIGNAL PREPROCESSING (Puja)
# ================================================================

class SignalPreprocessor:
    """
    Processes raw WESAD physiological signals into
    feature vectors for the Hybrid model.

    Pipeline:
        1. apply_bandpass_filter() — remove noise
        2. segment_signal()        — split into windows
        3. extract_time_features() — statistical features
        4. extract_freq_features() — FFT-based features
        5. extract_hrv_features()  — heart rate variability
        6. build_feature_vector()  — combine all features
        7. z_score_normalise()     — standardise to mean=0 std=1
    """

    def apply_bandpass_filter(
        self,
        signal_data  : np.ndarray,
        sampling_rate: float,
        low_freq     : float,
        high_freq    : float,
        order        : int = 4,
    ) -> np.ndarray:
        """
        Remove noise outside physiological frequency range.

        Recommended ranges (WESAD paper):
            ECG : 0.5 Hz to 45.0 Hz
            EDA : 0.05 Hz to 5.0 Hz

        Uses Butterworth bandpass filter applied
        forward and backward (filtfilt) to avoid phase shift.
        """
        nyquist  = sampling_rate / 2.0
        low      = low_freq  / nyquist
        high     = high_freq / nyquist
        b, a     = scipy_signal.butter(order, [low, high], btype="band")
        return scipy_signal.filtfilt(b, a, signal_data)

    def segment_signal(
        self,
        signal_data  : np.ndarray,
        sampling_rate: float,
    ) -> List[np.ndarray]:
        """
        Split long recording into 60-second overlapping windows.

        30-second overlap ensures stress events are not
        accidentally split across two windows.
        """
        window_size = int(WINDOW_SECONDS  * sampling_rate)
        step_size   = int(OVERLAP_SECONDS * sampling_rate)
        windows     = []
        start       = 0

        while start + window_size <= len(signal_data):
            windows.append(signal_data[start: start + window_size])
            start += step_size

        return windows

    def extract_time_features(
        self,
        window: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Calculate statistical features from one signal window.

        Features: mean, std, min, max, range, rms, peak count.
        Even simple stats like EDA mean/std can classify
        stress with ~75% accuracy in isolation.
        """
        peaks, _ = scipy_signal.find_peaks(window, height=0)
        return {
            f"{prefix}mean" : float(np.mean(window)),
            f"{prefix}std"  : float(np.std(window)),
            f"{prefix}min"  : float(np.min(window)),
            f"{prefix}max"  : float(np.max(window)),
            f"{prefix}range": float(np.max(window) - np.min(window)),
            f"{prefix}rms"  : float(np.sqrt(np.mean(window ** 2))),
            f"{prefix}peaks": float(len(peaks)),
        }

    def extract_freq_features(
        self,
        window       : np.ndarray,
        sampling_rate: float,
        prefix       : str = "",
    ) -> Dict[str, float]:
        """
        FFT-based frequency domain features.

        LF/HF ratio is a validated stress biomarker:
            LF band: 0.04–0.15 Hz
            HF band: 0.15–0.40 Hz
        Higher LF/HF ratio = more stress.
        """
        n         = len(window)
        fft_power = np.abs(np.fft.rfft(window)) ** 2
        freqs     = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

        lf_power = float(np.sum(fft_power[(freqs >= 0.04) & (freqs < 0.15)]))
        hf_power = float(np.sum(fft_power[(freqs >= 0.15) & (freqs < 0.40)]))
        ratio    = lf_power / hf_power if hf_power > 1e-6 else 0.0

        return {
            f"{prefix}lf_power"    : lf_power,
            f"{prefix}hf_power"    : hf_power,
            f"{prefix}lf_hf_ratio" : ratio,
        }

    def extract_hrv_features(
        self,
        ecg_window   : np.ndarray,
        sampling_rate: float,
    ) -> Dict[str, float]:
        """
        Heart Rate Variability from ECG R-peaks.

        HIGH HRV = relaxed/healthy
        LOW  HRV = stressed/anxious

        Metrics:
            mean_rr : Average R-R interval (ms)
            rmssd   : Root Mean Square of Successive Differences
            sdnn    : Standard Deviation of NN intervals
        """
        min_dist = int(0.3 * sampling_rate)
        r_peaks, _ = scipy_signal.find_peaks(
            ecg_window,
            distance = min_dist,
            height   = np.mean(ecg_window) + 0.5 * np.std(ecg_window),
        )

        if len(r_peaks) < 3:
            return {"hrv_mean_rr": 0.0, "hrv_rmssd": 0.0, "hrv_sdnn": 0.0}

        rr_ms     = np.diff(r_peaks) / sampling_rate * 1000.0
        succ_diff = np.diff(rr_ms)

        return {
            "hrv_mean_rr": float(np.mean(rr_ms)),
            "hrv_rmssd"  : float(np.sqrt(np.mean(succ_diff ** 2))),
            "hrv_sdnn"   : float(np.std(rr_ms)),
        }

    def build_feature_vector(
        self,
        ecg_window : np.ndarray,
        eda_window : np.ndarray,
        temp_window: np.ndarray,
    ) -> np.ndarray:
        """
        Combine all signal features into one flat array.

        Total features (~27):
            ECG  — 7 time + 3 freq + 3 HRV = 13
            EDA  — 7 time + 3 freq         = 10
            Temp — 4 time only              =  4

        Returns:
            1D numpy array, sorted by feature name for consistency
        """
        all_features = {}
        ecg_rate     = WESAD_SAMPLING_RATES["ecg"]
        eda_rate     = WESAD_SAMPLING_RATES["eda"]

        all_features.update(self.extract_time_features(ecg_window, "ecg_"))
        all_features.update(self.extract_freq_features(ecg_window, ecg_rate, "ecg_"))
        all_features.update(self.extract_hrv_features(ecg_window, ecg_rate))
        all_features.update(self.extract_time_features(eda_window, "eda_"))
        all_features.update(self.extract_freq_features(eda_window, eda_rate, "eda_"))

        temp_feats = self.extract_time_features(temp_window, "temp_")
        for key in ["temp_mean", "temp_std", "temp_range", "temp_rms"]:
            all_features[key] = temp_feats.get(key, 0.0)

        return np.array([all_features[k] for k in sorted(all_features.keys())])

    def z_score_normalise(
        self,
        feature_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardise features to mean=0, std=1.

        IMPORTANT: compute means/stds on TRAINING data only,
        then apply same values to validation and test sets.
        This prevents data leakage from future samples.

        Returns:
            normalised matrix, means array, stds array
        """
        means = np.mean(feature_matrix, axis=0)
        stds  = np.std(feature_matrix,  axis=0)
        stds[stds < 1e-8] = 1.0
        return (feature_matrix - means) / stds, means, stds


# ================================================================
#  SECTION 5 — LSTM MODEL (Puja)
# ================================================================

def build_lstm_model():
    """
    Build LSTM model for mental health text classification.

    Architecture:
        Embedding  → word tokens to 128-dim dense vectors
        LSTM-1     → 128 units, reads full sequence
        Dropout    → 0.3 (prevents overfitting)
        LSTM-2     → 64 units, returns final state
        Dropout    → 0.3
        Dense      → 32 units, ReLU
        Output     → 3 units, Softmax

    Why two LSTM layers:
        Layer 1 learns short-range word patterns
        Layer 2 combines them into sentence-level understanding

    Returns:
        Compiled Keras model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential(name="MindPulse_LSTM")

        model.add(tf.keras.layers.Embedding(
            input_dim    = VOCAB_SIZE,
            output_dim   = EMBEDDING_DIM,
            input_length = MAX_SEQUENCE_LENGTH,
            mask_zero    = True,
            name         = "token_embedding",
        ))
        model.add(tf.keras.layers.LSTM(128, return_sequences=True,  name="lstm_1"))
        model.add(tf.keras.layers.Dropout(0.3,                       name="dropout_1"))
        model.add(tf.keras.layers.LSTM(64,  return_sequences=False,  name="lstm_2"))
        model.add(tf.keras.layers.Dropout(0.3,                       name="dropout_2"))
        model.add(tf.keras.layers.Dense(32, activation="relu",       name="dense_hidden"))
        model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output"))

        model.compile(
            optimizer = Adam(learning_rate=LEARNING_RATE),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )
        return model

    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return None


def train_lstm(
    X_train   : np.ndarray,
    y_train   : np.ndarray,
    X_val     : np.ndarray,
    y_val     : np.ndarray,
    save_path : str = "models/saved/lstm_model.h5",
):
    """
    Train LSTM with early stopping and class weighting.

    Class weights fix imbalanced dataset:
        Without them model just predicts majority class.
        With them rare classes are penalised more on mistakes.

    Callbacks:
        EarlyStopping     — stop if val_loss stops improving (patience=5)
        ModelCheckpoint   — save best weights automatically
        ReduceLROnPlateau — halve learning rate when training stalls
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import (
            EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        classes       = np.unique(y_train)
        weights_arr   = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes.tolist(), weights_arr.tolist()))

        model = build_lstm_model()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(save_path, monitor="val_accuracy",
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs          = MAX_EPOCHS,
            batch_size      = BATCH_SIZE,
            class_weight    = class_weights,
            callbacks       = callbacks,
            verbose         = 1,
        )

        return model, history.history

    except ImportError:
        print("TensorFlow not installed.")
        return None, {}


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model") -> Dict:
    """
    Evaluate any classification model on test set.

    Reports:
        Accuracy, F1 macro, F1 weighted, confusion matrix,
        per-class precision/recall/F1

    Args:
        model      : Trained model with .predict() method
        X_test     : Test features
        y_test     : True labels
        model_name : Name for display

    Returns:
        Dictionary of all metrics
    """
    try:
        proba  = model.predict(X_test, verbose=0)
        y_pred = np.argmax(proba, axis=1)
    except Exception:
        y_pred = model.predict(X_test)
        proba  = None

    acc         = accuracy_score(y_test, y_pred)
    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    conf_mat    = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"{model_name} — TEST RESULTS")
    print(f"{'='*55}")
    print(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Macro         : {f1_macro:.4f}")
    print(f"F1 Weighted      : {f1_weighted:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")
    print(f"\nConfusion Matrix:\n{conf_mat}")
    print(f"{'='*55}")

    return {
        "accuracy"   : acc,
        "f1_macro"   : f1_macro,
        "f1_weighted": f1_weighted,
        "confusion"  : conf_mat,
        "y_pred"     : y_pred,
        "y_proba"    : proba,
    }


# ================================================================
#  SECTION 6 — BERT MODEL (Hanzla)
# ================================================================

def setup_bert_training(
    texts_train : List[str],
    labels_train: List[int],
    texts_val   : List[str],
    labels_val  : List[int],
    save_path   : str = "models/saved/bert_mindpulse",
):
    """
    Fine-tune BERT on mental health dataset.

    Fine-tuning strategy:
        1. Load bert-base-uncased (pre-trained on English text)
        2. Freeze all layers for first 3 epochs
        3. Unfreeze and fine-tune for remaining epochs
        4. Add classification head: 768 → 256 → 3

    Why bert-base-uncased:
        Already understands English language deeply.
        Fine-tuning on SMHD teaches it mental health language.
        Much faster and better than training from scratch.

    Run this on Google Colab with GPU for best results.
    Free GPU: colab.research.google.com

    Args:
        texts_train  : List of training Reddit posts
        labels_train : List of training labels (0/1/2)
        texts_val    : List of validation posts
        labels_val   : List of validation labels
        save_path    : Directory to save fine-tuned model
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            BertTokenizer,
            BertForSequenceClassification,
            AdamW,
            get_linear_schedule_with_warmup,
        )

        BERT_LR      = 2e-5
        BERT_EPOCHS  = 5
        BERT_BATCH   = 16
        BERT_MAXLEN  = 512
        DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {DEVICE}")
        print(f"Loading bert-base-uncased...")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model     = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=NUM_CLASSES
        ).to(DEVICE)

        class MHDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts  = texts
                self.labels = labels

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = tokenizer(
                    str(self.texts[idx]),
                    max_length     = BERT_MAXLEN,
                    padding        = "max_length",
                    truncation     = True,
                    return_tensors = "pt",
                )
                return {
                    "input_ids"     : enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "label"         : torch.tensor(self.labels[idx], dtype=torch.long),
                }

        train_loader = DataLoader(MHDataset(texts_train, labels_train),
                                  batch_size=BERT_BATCH, shuffle=True)
        val_loader   = DataLoader(MHDataset(texts_val, labels_val),
                                  batch_size=BERT_BATCH, shuffle=False)

        optimizer    = AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
        total_steps  = len(train_loader) * BERT_EPOCHS
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = total_steps // 10,
            num_training_steps = total_steps,
        )

        best_acc = 0.0
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(BERT_EPOCHS):
            model.train()
            total_loss = 0

            for batch in train_loader:
                ids   = batch["input_ids"].to(DEVICE)
                mask  = batch["attention_mask"].to(DEVICE)
                lbls  = batch["label"].to(DEVICE)

                optimizer.zero_grad()
                out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
                loss = out.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in val_loader:
                    ids  = batch["input_ids"].to(DEVICE)
                    mask = batch["attention_mask"].to(DEVICE)
                    lbls = batch["label"].to(DEVICE)
                    out  = model(input_ids=ids, attention_mask=mask)
                    pred = torch.argmax(out.logits, dim=1)
                    preds.extend(pred.cpu().numpy())
                    trues.extend(lbls.cpu().numpy())

            acc = accuracy_score(trues, preds)
            f1  = f1_score(trues, preds, average="macro")
            print(f"Epoch {epoch+1}/{BERT_EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  Saved best model (acc={acc:.4f})")

        print(f"\nBERT fine-tuning complete. Best acc: {best_acc:.4f}")
        return model, tokenizer

    except ImportError:
        print("Install: pip install torch transformers")
        return None, None


def predict_bert(
    text      : str,
    model_path: str = "models/saved/bert_mindpulse",
) -> Dict:
    """
    Run BERT inference on a single text string.

    Args:
        text       : Raw text string to classify
        model_path : Path to saved fine-tuned BERT model

    Returns:
        {"label": str, "confidence": float, "probabilities": dict}
    """
    try:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification

        DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model     = BertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        model.eval()

        enc = tokenizer(
            text,
            max_length     = 512,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

        with torch.no_grad():
            out   = model(
                input_ids      = enc["input_ids"].to(DEVICE),
                attention_mask = enc["attention_mask"].to(DEVICE),
            )
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]

        label_idx = int(np.argmax(probs))
        return {
            "label"        : CLASS_NAMES[label_idx].lower(),
            "confidence"   : float(probs[label_idx]),
            "probabilities": {
                "control"   : float(probs[0]),
                "depression": float(probs[1]),
                "anxiety"   : float(probs[2]),
            },
        }

    except Exception as e:
        return {"label": "control", "confidence": 0.5,
                "probabilities": {"control": 0.5, "depression": 0.3, "anxiety": 0.2},
                "error": str(e)}


# ================================================================
#  SECTION 7 — DATABASE (Ahmad)
# ================================================================

Base         = declarative_base()
engine       = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    """Stores user accounts. Passwords always hashed, never plain text."""
    __tablename__ = "users"

    user_id       = Column(String(36), primary_key=True,
                           default=lambda: str(uuid.uuid4()))
    username      = Column(String(50),  unique=True, nullable=False)
    email         = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at    = Column(DateTime,    default=datetime.utcnow)
    consent_given = Column(Boolean,     default=False)

    sessions = relationship("Session", back_populates="user",
                            cascade="all, delete-orphan")


class Session(Base):
    """One assessment session per user visit."""
    __tablename__ = "sessions"

    session_id   = Column(String(36), primary_key=True,
                          default=lambda: str(uuid.uuid4()))
    user_id      = Column(String(36), ForeignKey("users.user_id"), nullable=False)
    started_at   = Column(DateTime,   default=datetime.utcnow)
    completed_at = Column(DateTime,   nullable=True)
    input_mode   = Column(String(20), nullable=False)

    user        = relationship("User",        back_populates="sessions")
    predictions = relationship("Prediction",  back_populates="session",
                               cascade="all, delete-orphan")


class Prediction(Base):
    """Stores model prediction results."""
    __tablename__ = "predictions"

    prediction_id      = Column(String(36),  primary_key=True,
                                default=lambda: str(uuid.uuid4()))
    session_id         = Column(String(36),  ForeignKey("sessions.session_id"),
                                nullable=False)
    model_used         = Column(String(20),  nullable=False)
    label              = Column(String(20),  nullable=False)
    confidence         = Column(Float,       nullable=False)
    probability_vector = Column(JSON,        nullable=True)
    generated_at       = Column(DateTime,    default=datetime.utcnow)

    session = relationship("Session", back_populates="predictions")


class Database:
    """
    Simple database interface — CRUD operations for MindPulse.

    Usage:
        db = Database()
        db.create_tables()
        user = db.create_user("alice", "alice@email.com", "password")
        pred = db.save_prediction(session_id, "bert", "depression", 0.87)
    """

    def create_tables(self):
        """Create all database tables. Safe to run multiple times."""
        Base.metadata.create_all(bind=engine)

    def hash_password(self, password: str) -> str:
        """Hash password with random salt. Never store plain passwords."""
        salt   = secrets.token_hex(16)
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{hashed}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against its stored hash."""
        salt, hashed = stored_hash.split(":", 1)
        return hashlib.sha256((password + salt).encode()).hexdigest() == hashed

    def create_user(self, username: str, email: str,
                    password: str, consent: bool = False) -> Optional[User]:
        """Create a new user account."""
        db = SessionLocal()
        try:
            user = User(
                username      = username,
                email         = email.lower().strip(),
                password_hash = self.hash_password(password),
                consent_given = consent,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return user
        except Exception as e:
            db.rollback()
            return None
        finally:
            db.close()

    def get_user(self, username: str) -> Optional[User]:
        """Retrieve user by username."""
        db = SessionLocal()
        try:
            return db.query(User).filter(User.username == username).first()
        finally:
            db.close()

    def create_session(self, user_id: str, input_mode: str) -> Session:
        """Start a new assessment session."""
        db = SessionLocal()
        try:
            sess = Session(user_id=user_id, input_mode=input_mode)
            db.add(sess)
            db.commit()
            db.refresh(sess)
            return sess
        finally:
            db.close()

    def save_prediction(
        self,
        session_id : str,
        model_used : str,
        label      : str,
        confidence : float,
        proba_vec  : Optional[Dict] = None,
    ) -> Prediction:
        """Save a model prediction to the database."""
        db = SessionLocal()
        try:
            pred = Prediction(
                session_id         = session_id,
                model_used         = model_used,
                label              = label,
                confidence         = round(float(confidence), 4),
                probability_vector = proba_vec or {},
            )
            db.add(pred)
            db.commit()
            db.refresh(pred)
            return pred
        finally:
            db.close()

    def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's prediction history, newest first."""
        db = SessionLocal()
        try:
            results = (
                db.query(Prediction, Session)
                .join(Session, Prediction.session_id == Session.session_id)
                .filter(Session.user_id == user_id)
                .order_by(Prediction.generated_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "date"         : pred.generated_at.strftime("%Y-%m-%d %H:%M"),
                    "model"        : pred.model_used,
                    "label"        : pred.label,
                    "confidence"   : f"{pred.confidence*100:.1f}%",
                    "input_mode"   : sess.input_mode,
                    "probabilities": pred.probability_vector,
                }
                for pred, sess in results
            ]
        finally:
            db.close()


# ================================================================
#  SECTION 8 — STREAMLIT APPLICATION (Usama)
# ================================================================

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title            = "MindPulse",
    page_icon             = "🧠",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .ethics-banner {
        background: #FAEEDA;
        border-left: 4px solid #854F0B;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 14px;
        color: #412402;
    }
    .result-card {
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .metric-tile {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models (once at startup) ────────────────────────────
@st.cache_resource
def load_all_models():
    """
    Load all trained models into memory once.

    @st.cache_resource ensures this runs only ONCE —
    not on every user interaction. Critical for BERT
    which is ~400MB and would be unbearably slow to
    reload on every button click.
    """
    models = {"lstm": None, "bert": None, "hybrid": None}

    try:
        import tensorflow as tf
        models["lstm"] = tf.keras.models.load_model("models/saved/lstm_model.h5")
    except Exception:
        pass

    try:
        from transformers import pipeline
        models["bert"] = pipeline(
            "text-classification",
            model             = "models/saved/bert_mindpulse",
            return_all_scores = True,
        )
    except Exception:
        pass

    return models


# ── Initialise database ───────────────────────────────────────
@st.cache_resource
def get_database():
    db = Database()
    db.create_tables()
    return db


# ── Helper: ethics banner ─────────────────────────────────────
def show_ethics_banner():
    """Mandatory ethics disclaimer — shown on every results page."""
    st.markdown("""
    <div class="ethics-banner">
        ⚠️ <strong>Important:</strong> MindPulse is a <strong>screening tool only</strong>
        — NOT a medical diagnosis. Always consult a qualified mental health professional.
        Crisis support: <strong>Samaritans 116 123</strong> (free, 24/7).
    </div>
    """, unsafe_allow_html=True)


# ── Helper: show prediction result ───────────────────────────
def show_prediction_result(
    label        : str,
    confidence   : float,
    probabilities: Dict,
):
    """Display model prediction in a styled card with recommendations."""
    icons = {"depression": "😞", "anxiety": "😰",
             "stress": "😤", "control": "✅"}
    color = COLORS.get(label, "#5F5E5A")
    icon  = icons.get(label, "🔍")

    st.markdown(f"""
    <div class="result-card" style="background:{color}15;border:1.5px solid {color}">
        <div style="font-size:26px;font-weight:500;color:{color}">
            {icon} {label.title()} Detected
        </div>
        <div style="font-size:14px;color:{color};margin-top:4px">
            Confidence: {confidence*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Probability Breakdown")
    for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(cls.title())
        with c2:
            st.progress(float(prob), text=f"{float(prob)*100:.1f}%")

    # Recommendations
    st.write("")
    st.subheader("What to do next")
    recs = {
        "depression": [
            "Speak to your GP or a mental health professional",
            "Mind UK: 0300 123 3393 | mind.org.uk",
            "Samaritans (24/7): 116 123",
        ],
        "anxiety": [
            "Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s",
            "Anxiety UK: 03444 775 774 | anxietyuk.org.uk",
            "Samaritans (24/7): 116 123",
        ],
        "stress": [
            "Your university counselling service is free — use it",
            "Student Minds: studentminds.org.uk",
            "Samaritans (24/7): 116 123",
        ],
        "control": [
            "Your results suggest no significant concern right now",
            "Continue monitoring how you feel over time",
            "It is always okay to seek support if things change",
        ],
    }
    for rec in recs.get(label, []):
        st.markdown(f"• {rec}")


# ── PAGE: Home ────────────────────────────────────────────────
def page_home():
    st.title("🧠 MindPulse")
    st.subheader("AI Mental Wellbeing Screening System")
    show_ethics_banner()

    st.markdown("""
    MindPulse uses AI to screen for signs of **depression**, **anxiety**, and **stress**.
    Choose how you want to be assessed below.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📝 **Text Journal**\n\nWrite how you feel — BERT & LSTM analyse it")
        if st.button("Start Text Analysis", use_container_width=True):
            st.session_state["page"] = "text"
    with c2:
        st.info("📋 **DASS-21 Quiz**\n\nAnswer 21 questions — takes 3 minutes")
        if st.button("Take the Quiz", use_container_width=True):
            st.session_state["page"] = "quiz"
    with c3:
        st.info("📈 **Physio Upload**\n\nUpload biosensor CSV — Hybrid model")
        if st.button("Upload Signals", use_container_width=True):
            st.session_state["page"] = "physio"


# ── PAGE: Text Analysis ───────────────────────────────────────
def page_text_analysis(models: dict, db: Database):
    st.title("📝 Text Analysis")
    show_ethics_banner()
    st.write("Describe how you have been feeling recently.")

    text_input   = st.text_area(
        "Your journal entry",
        placeholder      = "I have been feeling...",
        height           = 200,
        max_chars        = 2000,
        label_visibility = "collapsed",
    )
    model_choice = st.radio(
        "Choose model",
        options    = ["BERT (recommended)", "LSTM", "Both — ensemble"],
        horizontal = True,
    )

    if st.button("Analyse Text", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please write something before analysing.")
            return
        if len(text_input.split()) < 10:
            st.warning("Please write at least 10 words for a meaningful result.")
            return

        with st.spinner("Analysing your text..."):
            # Try real BERT model first
            if models.get("bert") and "BERT" in model_choice:
                result = predict_bert(text_input)
                label        = result["label"]
                confidence   = result["confidence"]
                probabilities = result["probabilities"]
                model_used   = "bert"
            else:
                # Placeholder until models are trained
                label         = "anxiety"
                confidence    = 0.74
                probabilities = {
                    "control"   : 0.09,
                    "depression": 0.17,
                    "anxiety"   : 0.74,
                }
                model_used = "placeholder"

        show_prediction_result(label, confidence, probabilities)


# ── PAGE: DASS-21 Quiz ────────────────────────────────────────
def page_quiz(db: Database):
    st.title("📋 DASS-21 Questionnaire")
    show_ethics_banner()

    st.write("""
    Rate each statement over the **past week**:
    **0** = Never | **1** = Sometimes | **2** = Often | **3** = Almost always
    """)

    questions = {
        "q1" : "I found it hard to wind down",
        "q2" : "I was aware of dryness of my mouth",
        "q3" : "I couldn't seem to experience any positive feeling at all",
        "q4" : "I experienced breathing difficulty",
        "q5" : "I found it difficult to work up the initiative to do things",
        "q6" : "I tended to over-react to situations",
        "q7" : "I experienced trembling (e.g. in the hands)",
        "q8" : "I felt that I was using a lot of nervous energy",
        "q9" : "I was worried about situations in which I might panic",
        "q10": "I felt that I had nothing to look forward to",
        "q11": "I found myself getting agitated",
        "q12": "I found it difficult to relax",
        "q13": "I felt down-hearted and blue",
        "q14": "I was intolerant of anything that kept me from getting on",
        "q15": "I felt I was close to panic",
        "q16": "I was unable to become enthusiastic about anything",
        "q17": "I felt I wasn't worth much as a person",
        "q18": "I felt that I was rather touchy",
        "q19": "I was aware of the action of my heart without physical exertion",
        "q20": "I felt scared without any good reason",
        "q21": "I felt that life was meaningless",
    }

    options = {
        "0 — Never"       : 0,
        "1 — Sometimes"   : 1,
        "2 — Often"       : 2,
        "3 — Almost always": 3,
    }
    answers = {}

    with st.form("dass21_form"):
        for q_key, q_text in questions.items():
            q_num    = int(q_key[1:])
            st.write(f"**{q_num}. {q_text}**")
            selected = st.radio(
                label            = q_text,
                options          = list(options.keys()),
                horizontal       = True,
                key              = f"dass_{q_key}",
                label_visibility = "collapsed",
            )
            answers[q_key] = options[selected]
            st.divider()

        submitted = st.form_submit_button(
            "Get My Results",
            type                = "primary",
            use_container_width = True,
        )

    if submitted:
        scores           = compute_subscale_scores(answers)
        label, confidence = classify_dass_label(scores)

        st.write("**Your subscale scores:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Depression", scores["depression"])
        with c2:
            st.metric("Anxiety",    scores["anxiety"])
        with c3:
            st.metric("Stress",     scores["stress"])

        total         = max(sum(scores.values()), 1)
        probabilities = {k: round(v / total, 3) for k, v in scores.items()}
        show_prediction_result(label, confidence, probabilities)


# ── PAGE: Physiological Upload ────────────────────────────────
def page_physio():
    st.title("📈 Physiological Signal Upload")
    show_ethics_banner()

    st.write("Upload a CSV file from your wearable device (ECG, EDA, temperature).")
    st.info("Expected columns: `ecg`, `eda`, `temperature` — one row per sample")

    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"File received: {uploaded.name}")
            st.write("**Data preview (first 5 rows):**")
            st.dataframe(df.head())
            st.write(f"Total rows: {len(df):,} | Columns: {list(df.columns)}")

            if st.button("Run Signal Analysis", type="primary"):
                with st.spinner("Processing physiological signals..."):
                    sp = SignalPreprocessor()

                    # Try to extract real features if correct columns exist
                    if all(col in df.columns for col in ["ecg", "eda", "temperature"]):
                        ecg_rate = WESAD_SAMPLING_RATES["ecg"]
                        eda_rate = WESAD_SAMPLING_RATES["eda"]

                        ecg_filtered = sp.apply_bandpass_filter(
                            df["ecg"].values, ecg_rate, 0.5, 45.0
                        )
                        ecg_wins = sp.segment_signal(ecg_filtered, ecg_rate)
                        eda_wins = sp.segment_signal(df["eda"].values, eda_rate)
                        tmp_wins = sp.segment_signal(df["temperature"].values, eda_rate)

                        if ecg_wins and eda_wins and tmp_wins:
                            features = sp.build_feature_vector(
                                ecg_wins[0], eda_wins[0], tmp_wins[0]
                            )
                            st.write(f"Extracted {len(features)} physiological features")

                    # Placeholder result — replace with real Hybrid model
                    label         = "stress"
                    confidence    = 0.81
                    probabilities = {
                        "control"   : 0.06,
                        "depression": 0.13,
                        "anxiety"   : 0.00,
                        "stress"    : 0.81,
                    }

                show_prediction_result(label, confidence, probabilities)

        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.write("Please make sure it is a valid CSV file.")


# ── PAGE: History ─────────────────────────────────────────────
def page_history(db: Database):
    st.title("📊 My History")

    # Placeholder history — replace with db.get_user_history(user_id)
    history = [
        {"Date": "2025-07-10", "Mode": "Text",          "Result": "Anxiety",    "Confidence": "74%"},
        {"Date": "2025-07-07", "Mode": "Questionnaire", "Result": "Anxiety",    "Confidence": "81%"},
        {"Date": "2025-07-01", "Mode": "Physiological", "Result": "Stress",     "Confidence": "68%"},
        {"Date": "2025-06-22", "Mode": "Text",          "Result": "Depression", "Confidence": "61%"},
        {"Date": "2025-06-15", "Mode": "Questionnaire", "Result": "Control",    "Confidence": "88%"},
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Sessions", len(history))
    with c2:
        st.metric("Most Recent", history[0]["Result"])

    st.write("")
    st.subheader("Session Log")
    st.dataframe(pd.DataFrame(history), use_container_width=True)

    st.subheader("Confidence Trend")
    chart_data = pd.DataFrame({
        "Session"   : list(range(1, len(history) + 1)),
        "Confidence": [int(h["Confidence"].replace("%", "")) for h in history],
    })
    st.line_chart(chart_data.set_index("Session"))

    if st.button("Export History as CSV"):
        csv = pd.DataFrame(history).to_csv(index=False)
        st.download_button(
            label     = "Download CSV",
            data      = csv,
            file_name = "mindpulse_history.csv",
            mime      = "text/csv",
        )


# ── PAGE: About & Ethics ──────────────────────────────────────
def page_about():
    st.title("ℹ️ About MindPulse & Ethics")

    st.subheader("What is MindPulse?")
    st.write("""
    MindPulse is a university research project that uses AI to screen
    for signs of depression, anxiety, and stress. It is NOT a clinical
    diagnostic tool and should never replace professional medical advice.
    """)

    st.subheader("Datasets Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **SMHD** — Reddit posts with self-reported mental health diagnoses

        **DASS-21** — Depression Anxiety Stress Scale questionnaire responses
        """)
    with col2:
        st.markdown("""
        **WESAD** — Wearable physiological sensor recordings

        **Custom** — Simulated dataset for data augmentation
        """)

    st.subheader("Models")
    st.markdown("""
    | Model | Input | Purpose |
    |-------|-------|---------|
    | LSTM | Text sequences | Pattern recognition in text |
    | BERT (fine-tuned) | Tokenised text | Deep language understanding |
    | Hybrid | Text + physiological | Multi-modal fusion |
    """)

    st.subheader("Ethics Statement")
    st.error("""
    ⚠️ MindPulse is a SCREENING TOOL only — not a medical diagnosis.

    • Results should never replace professional clinical assessment
    • If you are in crisis, please contact Samaritans: 116 123 (free, 24/7)
    • All data is stored securely and never shared with third parties
    • You may delete your data at any time
    • This system may have biases — it was trained on English Reddit data
      and may not perform equally for all populations
    """)

    st.subheader("Team")
    team = {
        "Sahar"  : "Group Leader",
        "Ahmad"  : "Deputy Leader & Database",
        "Puja"   : "ML Engineer — LSTM & Hybrid",
        "Hanzla" : "NLP Specialist & QA Lead — BERT",
        "Usama"  : "App Developer — Streamlit",
        "Irfan"  : "Ethics & Bias Lead",
    }
    for name, role in team.items():
        st.markdown(f"**{name}** — {role}")


# ================================================================
#  SECTION 9 — MAIN APP ROUTER (Usama)
# ================================================================

def main():
    """
    Main function — sets up sidebar navigation and routes
    user to the correct page based on their selection.

    Session state tracks which page is active so Streamlit
    does not reset to home on every interaction.
    """
    # Initialise session state
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    # Load resources (cached — only runs once)
    models = load_all_models()
    db     = get_database()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧠 MindPulse")
        st.caption("AI Mental Wellbeing System")
        st.divider()

        nav_pages = {
            "🏠 Home"           : "home",
            "📝 Text Analysis"  : "text",
            "📋 DASS-21 Quiz"   : "quiz",
            "📈 Physio Upload"  : "physio",
            "📊 My History"     : "history",
            "ℹ️ About & Ethics" : "about",
        }

        for label, key in nav_pages.items():
            if st.button(label, use_container_width=True):
                st.session_state["page"] = key

        st.divider()

        # Model status indicators
        st.caption("Model Status")
        st.markdown(
            "🟢 LSTM" if models.get("lstm") else "🔴 LSTM (not trained)"
        )
        st.markdown(
            "🟢 BERT" if models.get("bert") else "🔴 BERT (not trained)"
        )
        st.markdown("🔴 Hybrid (coming soon)")
        st.divider()
        st.caption("Not a medical tool")
        st.caption("v1.0 — University Project")

    # Route to page
    page = st.session_state.get("page", "home")

    if page == "home":
        page_home()
    elif page == "text":
        page_text_analysis(models, db)
    elif page == "quiz":
        page_quiz(db)
    elif page == "physio":
        page_physio()
    elif page == "history":
        page_history(db)
    elif page == "about":
        page_about()


# ================================================================
#  SECTION 10 — UNIT TESTS (Hanzla)
# ================================================================

import pytest


class TestTextPreprocessor:
    """Unit tests for text preprocessing functions."""

    @pytest.fixture
    def tp(self):
        return TextPreprocessor()

    def test_clean_removes_url(self, tp):
        assert "http" not in tp.clean_text("visit http://example.com today please")

    def test_clean_removes_mention(self, tp):
        assert "helpfuluser" not in tp.clean_text("thanks u/helpfuluser for advice")

    def test_clean_removes_subreddit(self, tp):
        assert "depression" not in tp.clean_text("posted on r/depression yesterday morning")

    def test_clean_lowercase(self, tp):
        result = tp.clean_text("I Feel VERY Anxious TODAY")
        assert result == result.lower()

    def test_clean_none_returns_empty(self, tp):
        assert tp.clean_text(None) == ""

    def test_clean_empty_returns_empty(self, tp):
        assert tp.clean_text("") == ""

    def test_valid_post_long(self, tp):
        assert tp.is_valid_post("word " * 25) is True

    def test_valid_post_short(self, tp):
        assert tp.is_valid_post("too short") is False

    def test_pad_short_sequence(self, tp):
        padded = tp.pad_sequence([1, 2, 3])
        assert len(padded) == MAX_SEQUENCE_LENGTH

    def test_pad_long_sequence(self, tp):
        padded = tp.pad_sequence(list(range(MAX_SEQUENCE_LENGTH + 50)))
        assert len(padded) == MAX_SEQUENCE_LENGTH

    def test_pad_returns_numpy(self, tp):
        assert isinstance(tp.pad_sequence([1, 2, 3]), np.ndarray)

    def test_encode_raises_without_vocab(self, tp):
        with pytest.raises(RuntimeError):
            tp.encode("some text here today and more words")

    def test_vocab_pad_at_zero(self, tp):
        tp.build_vocab(["hello world today feeling good very much"])
        assert tp.word_to_index[PADDING_TOKEN] == 0

    def test_vocab_unk_at_one(self, tp):
        tp.build_vocab(["hello world today feeling good very much"])
        assert tp.word_to_index[UNKNOWN_TOKEN] == 1


class TestDASS21:
    """Unit tests for DASS-21 scoring functions."""

    def test_all_zeros_gives_zero(self):
        answers = {f"q{i}": 0 for i in range(1, 22)}
        scores  = compute_subscale_scores(answers)
        assert scores["depression"] == 0
        assert scores["anxiety"]    == 0
        assert scores["stress"]     == 0

    def test_all_threes_gives_max(self):
        answers = {f"q{i}": 3 for i in range(1, 22)}
        scores  = compute_subscale_scores(answers)
        assert scores["depression"] == 42
        assert scores["anxiety"]    == 42
        assert scores["stress"]     == 42

    def test_depression_threshold(self):
        answers = {f"q{i}": 0 for i in range(1, 22)}
        for q in DEPRESSION_ITEMS:
            answers[f"q{q}"] = 1
        scores = compute_subscale_scores(answers)
        label, _ = classify_dass_label(scores)
        assert label == "depression"

    def test_anxiety_threshold(self):
        answers = {f"q{i}": 0 for i in range(1, 22)}
        for q in ANXIETY_ITEMS:
            answers[f"q{q}"] = 1
        scores = compute_subscale_scores(answers)
        label, _ = classify_dass_label(scores)
        assert label == "anxiety"

    def test_control_label(self):
        answers = {f"q{i}": 0 for i in range(1, 22)}
        scores  = compute_subscale_scores(answers)
        label, _ = classify_dass_label(scores)
        assert label == "control"

    def test_feature_vector_shape(self):
        answers = {f"q{i}": 1 for i in range(1, 22)}
        vec     = build_dass_feature_vector(answers)
        assert vec.shape == (24,)

    def test_normalised_between_zero_and_one(self):
        scores = {"depression": 20, "anxiety": 14, "stress": 18}
        norm   = normalise_dass_scores(scores)
        assert all(0.0 <= v <= 1.0 for v in norm.values())


class TestSignalPreprocessor:
    """Unit tests for physiological signal preprocessing."""

    @pytest.fixture
    def sp(self):
        return SignalPreprocessor()

    def test_filter_same_length(self, sp):
        raw      = np.random.randn(7000)
        filtered = sp.apply_bandpass_filter(raw, 700, 0.5, 45.0)
        assert len(filtered) == len(raw)

    def test_filter_no_nan(self, sp):
        filtered = sp.apply_bandpass_filter(np.random.randn(700), 700, 0.5, 45.0)
        assert not np.isnan(filtered).any()

    def test_segmentation_count(self, sp):
        signal = np.random.randn(120 * 700)
        wins   = sp.segment_signal(signal, 700)
        assert len(wins) == 3

    def test_short_signal_no_windows(self, sp):
        signal = np.random.randn(10 * 700)
        wins   = sp.segment_signal(signal, 700)
        assert len(wins) == 0

    def test_time_features_keys(self, sp):
        feats = sp.extract_time_features(np.random.randn(700), "ecg_")
        for key in ["ecg_mean", "ecg_std", "ecg_min", "ecg_max",
                    "ecg_range", "ecg_rms", "ecg_peaks"]:
            assert key in feats

    def test_feature_vector_consistent_length(self, sp):
        v1 = sp.build_feature_vector(
            np.random.randn(42000), np.random.randn(240), np.random.randn(240)
        )
        v2 = sp.build_feature_vector(
            np.random.randn(42000), np.random.randn(240), np.random.randn(240)
        )
        assert len(v1) == len(v2)

    def test_normalisation_shape(self, sp):
        matrix     = np.random.randn(100, 27)
        norm, _, _ = sp.z_score_normalise(matrix)
        assert norm.shape == matrix.shape

    def test_normalisation_no_nan(self, sp):
        matrix     = np.random.randn(50, 10)
        norm, _, _ = sp.z_score_normalise(matrix)
        assert not np.isnan(norm).any()


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
