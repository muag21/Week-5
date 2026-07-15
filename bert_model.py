# ============================================================
#  MindPulse — BERT Fine-tuning
#  Author  : Hanzla (NLP Specialist & QA Lead)
#  File    : models/bert_model.py
#  Purpose : Fine-tune BERT for mental health text
#            classification using HuggingFace Transformers
#            Labels: 0=Control, 1=Depression, 2=Anxiety
# ============================================================

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)
from typing import List, Dict, Tuple


# ── Configuration ─────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"   # Pre-trained model to fine-tune
MAX_LENGTH      = 512                    # Max tokens for BERT
BATCH_SIZE      = 16                     # Small batch — BERT is large
LEARNING_RATE   = 2e-5                   # Standard for BERT fine-tuning
NUM_EPOCHS      = 5                      # Fine-tune for 5 epochs max
NUM_CLASSES     = 3                      # depression / anxiety / control
CLASS_NAMES     = ["Control", "Depression", "Anxiety"]

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Dataset Class ─────────────────────────────────────────────
class MentalHealthDataset(Dataset):
    """
    PyTorch Dataset for mental health text classification.

    Takes a list of texts and labels, tokenises them using
    BERT tokeniser, and returns tensors ready for BERT.

    Args:
        texts     : List of raw text strings
        labels    : List of integer labels (0, 1, or 2)
        tokenizer : HuggingFace BERT tokenizer
        max_length: Maximum token length (512 for BERT)
    """

    def __init__(
        self,
        texts     : List[str],
        labels    : List[int],
        tokenizer : BertTokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenise the text
        encoding = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt"
        )

        return {
            "input_ids"      : encoding["input_ids"].squeeze(),
            "attention_mask" : encoding["attention_mask"].squeeze(),
            "label"          : torch.tensor(label, dtype=torch.long)
        }


# ── Load Tokenizer and Model ───────────────────────────────────
def load_bert_model(num_classes: int = NUM_CLASSES):
    """
    Load pre-trained BERT tokenizer and model.

    We use BertForSequenceClassification which adds a
    classification head on top of the pre-trained BERT encoder.

    Args:
        num_classes: Number of output classes (3 for MindPulse)

    Returns:
        tokenizer : BERT WordPiece tokenizer
        model     : BERT model with classification head
    """
    print(f"Loading {BERT_MODEL_NAME}...")

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels = num_classes,
    )

    model = model.to(DEVICE)
    print("BERT model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return tokenizer, model


# ── Train BERT ────────────────────────────────────────────────
def train_bert(
    texts_train  : List[str],
    labels_train : List[int],
    texts_val    : List[str],
    labels_val   : List[int],
    save_path    : str = "models/saved/bert_mindpulse",
) -> Tuple[BertForSequenceClassification, dict]:
    """
    Fine-tune BERT on the mental health dataset.

    Fine-tuning means:
        1. Start with BERT that already understands English
        2. Train it a little more on our mental health data
        3. It learns to recognise depression/anxiety language

    This is much better than training from scratch because:
        - BERT already understands language deeply
        - We only need a small dataset to fine-tune
        - Training is much faster

    Args:
        texts_train  : List of training text strings
        labels_train : List of training labels (0, 1, or 2)
        texts_val    : List of validation text strings
        labels_val   : List of validation labels
        save_path    : Directory to save the fine-tuned model

    Returns:
        fine-tuned model and training history dictionary
    """
    os.makedirs(save_path, exist_ok=True)

    # Load tokenizer and model
    tokenizer, model = load_bert_model()

    # Create datasets
    train_dataset = MentalHealthDataset(texts_train, labels_train, tokenizer)
    val_dataset   = MentalHealthDataset(texts_val,   labels_val,   tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer — AdamW is standard for BERT fine-tuning
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Learning rate scheduler — warms up then decreases
    total_steps    = len(train_loader) * NUM_EPOCHS
    warmup_steps   = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = 0.0

    print(f"\nFine-tuning BERT on {len(texts_train):,} samples...")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}\n")

    for epoch in range(NUM_EPOCHS):
        # ── Training phase ────────────────────────────────────
        model.train()
        total_train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation phase ──────────────────────────────────
        model.eval()
        total_val_loss = 0
        all_preds      = []
        all_labels     = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["label"].to(DEVICE)

                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels
                )

                total_val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss    = total_val_loss / len(val_loader)
        val_accuracy    = accuracy_score(all_labels, all_preds)
        val_f1          = f1_score(all_labels, all_preds, average="macro")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss    : {avg_train_loss:.4f}")
        print(f"  Val Loss      : {avg_val_loss:.4f}")
        print(f"  Val Accuracy  : {val_accuracy:.4f}")
        print(f"  Val F1 Macro  : {val_f1:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  New best model saved! Accuracy: {val_accuracy:.4f}")

    print(f"\nTraining complete! Best validation accuracy: {best_val_accuracy:.4f}")
    return model, history


# ── Evaluate BERT ─────────────────────────────────────────────
def evaluate_bert(
    model_path : str,
    texts_test : List[str],
    labels_test: List[int],
) -> Dict:
    """
    Evaluate the fine-tuned BERT model on test set.

    Args:
        model_path  : Path to saved fine-tuned model
        texts_test  : List of test text strings
        labels_test : List of true labels

    Returns:
        Dictionary of evaluation metrics
    """
    print("\nLoading fine-tuned BERT model...")

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model     = BertForSequenceClassification.from_pretrained(model_path)
    model     = model.to(DEVICE)
    model.eval()

    test_dataset = MentalHealthDataset(texts_test, labels_test, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)
            preds   = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc         = accuracy_score(all_labels, all_preds)
    f1_macro    = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    conf_mat    = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print("BERT MODEL — TEST RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Macro         : {f1_macro:.4f}")
    print(f"F1 Weighted      : {f1_weighted:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES)}")
    print(f"\nConfusion Matrix:\n{conf_mat}")
    print(f"{'='*50}")

    return {
        "accuracy"    : acc,
        "f1_macro"    : f1_macro,
        "f1_weighted" : f1_weighted,
        "confusion"   : conf_mat,
        "y_pred"      : all_preds,
        "y_proba"     : all_probs,
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing BERT model setup...")
    print(f"Device: {DEVICE}")
    print(f"BERT model: {BERT_MODEL_NAME}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class names: {CLASS_NAMES}")
    print("\nBERT setup ready for fine-tuning!")
    print("Run on Google Colab for GPU support.")
