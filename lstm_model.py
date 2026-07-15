# ============================================================
#  MindPulse — LSTM Model
#  Author  : Puja (ML Engineer)
#  File    : models/lstm_model.py
#  Purpose : Define, train and evaluate the LSTM model
#            for mental health text classification
#            Labels: 0=Control, 1=Depression, 2=Anxiety
# ============================================================

import numpy as np
import os
from typing import Tuple, Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight


# ── Configuration ─────────────────────────────────────────────
VOCAB_SIZE          = 20002   # vocabulary size + 2 special tokens
EMBEDDING_DIM       = 128     # each word → 128 dimensional vector
MAX_SEQUENCE_LENGTH = 256     # max tokens per post
NUM_CLASSES         = 3       # depression / anxiety / control
BATCH_SIZE          = 64
LEARNING_RATE       = 0.001
MAX_EPOCHS          = 50
CLASS_NAMES         = ["Control", "Depression", "Anxiety"]


# ── Build Model ───────────────────────────────────────────────
def build_lstm_model() -> tf.keras.Model:
    """
    Build the LSTM model for mental health classification.

    Architecture:
        Embedding  → converts word tokens to dense vectors
        LSTM-1     → 128 units, reads full sequence
        Dropout    → 0.3 (prevents overfitting)
        LSTM-2     → 64 units, returns final state only
        Dropout    → 0.3
        Dense      → 32 units, ReLU activation
        Output     → 3 units, Softmax (one per class)

    Returns:
        Compiled Keras model ready for training
    """
    model = Sequential(name="MindPulse_LSTM")

    # Embedding layer — converts integer tokens to dense vectors
    model.add(Embedding(
        input_dim    = VOCAB_SIZE,
        output_dim   = EMBEDDING_DIM,
        input_length = MAX_SEQUENCE_LENGTH,
        mask_zero    = True,
        name         = "token_embedding"
    ))

    # First LSTM layer — learns short range word patterns
    model.add(LSTM(
        units            = 128,
        return_sequences = True,
        name             = "lstm_layer_1"
    ))
    model.add(Dropout(0.3, name="dropout_1"))

    # Second LSTM layer — combines patterns into one vector
    model.add(LSTM(
        units            = 64,
        return_sequences = False,
        name             = "lstm_layer_2"
    ))
    model.add(Dropout(0.3, name="dropout_2"))

    # Dense hidden layer
    model.add(Dense(32, activation="relu", name="dense_hidden"))

    # Output layer — softmax gives probability for each class
    model.add(Dense(NUM_CLASSES, activation="softmax", name="output_layer"))

    # Compile the model
    model.compile(
        optimizer = Adam(learning_rate=LEARNING_RATE),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    return model


# ── Train Model ───────────────────────────────────────────────
def train_lstm(
    X_train   : np.ndarray,
    y_train   : np.ndarray,
    X_val     : np.ndarray,
    y_val     : np.ndarray,
    save_path : str = "models/saved/lstm_model.h5",
) -> Tuple[tf.keras.Model, dict]:
    """
    Train the LSTM model with early stopping and class weighting.

    Class weights handle imbalanced dataset — if depression has
    more samples than anxiety, the model would just predict
    depression all the time without class weights.

    Args:
        X_train   : Shape (N_train, MAX_SEQUENCE_LENGTH)
        y_train   : Shape (N_train,) — labels 0, 1, or 2
        X_val     : Shape (N_val, MAX_SEQUENCE_LENGTH)
        y_val     : Shape (N_val,) — labels 0, 1, or 2
        save_path : Where to save the best model

    Returns:
        trained model and training history dictionary
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Compute class weights to handle imbalance
    classes      = np.unique(y_train)
    weights_arr  = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes.tolist(), weights_arr.tolist()))
    print(f"Class weights: {class_weights}")

    # Build model
    model = build_lstm_model()
    print(model.summary())

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor              = "val_loss",
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1
        ),
        ModelCheckpoint(
            filepath       = save_path,
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 1
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-6,
            verbose  = 1
        )
    ]

    # Train
    print(f"\nTraining LSTM on {len(X_train):,} samples...")
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = MAX_EPOCHS,
        batch_size      = BATCH_SIZE,
        class_weight    = class_weights,
        callbacks       = callbacks,
        verbose         = 1
    )

    print(f"\nBest model saved to: {save_path}")
    return model, history.history


# ── Evaluate Model ────────────────────────────────────────────
def evaluate_lstm(
    model  : tf.keras.Model,
    X_test : np.ndarray,
    y_test : np.ndarray,
) -> Dict:
    """
    Evaluate the trained LSTM on the held-out test set.

    Reports:
        - Overall accuracy
        - F1 score per class
        - Macro and weighted F1
        - Confusion matrix

    Args:
        model  : Trained Keras model
        X_test : Shape (N_test, MAX_SEQUENCE_LENGTH)
        y_test : True labels shape (N_test,)

    Returns:
        Dictionary of all evaluation metrics
    """
    print("\nEvaluating LSTM on test set...")

    # Get predictions
    proba  = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(proba, axis=1)

    # Compute metrics
    acc         = np.mean(y_pred == y_test)
    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    conf_mat    = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print("LSTM MODEL — TEST RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Macro         : {f1_macro:.4f}")
    print(f"F1 Weighted      : {f1_weighted:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")
    print(f"\nConfusion Matrix:\n{conf_mat}")
    print(f"{'='*50}")

    return {
        "accuracy"    : acc,
        "f1_macro"    : f1_macro,
        "f1_weighted" : f1_weighted,
        "confusion"   : conf_mat,
        "y_pred"      : y_pred,
        "y_proba"     : proba,
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building LSTM model...")
    model = build_lstm_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print("\nModel built successfully!")
