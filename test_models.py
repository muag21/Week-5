# ============================================================
#  MindPulse — Model Unit Tests
#  Author  : Hanzla (NLP Specialist & QA Lead)
#  File    : tests/test_models.py
#  Run     : pytest tests/test_models.py -v
# ============================================================

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── LSTM Tests ────────────────────────────────────────────────
class TestLSTMModel:

    def test_model_builds_without_error(self):
        """LSTM model should build without crashing."""
        try:
            import tensorflow as tf
            from models.lstm_model import build_lstm_model
            model = build_lstm_model()
            assert model is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_model_output_shape(self):
        """LSTM output should be (batch_size, 3) — one prob per class."""
        try:
            import tensorflow as tf
            from models.lstm_model import build_lstm_model, MAX_SEQUENCE_LENGTH
            model    = build_lstm_model()
            dummy_X  = np.zeros((4, MAX_SEQUENCE_LENGTH), dtype=np.int32)
            output   = model.predict(dummy_X, verbose=0)
            assert output.shape == (4, 3)
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_output_probabilities_sum_to_one(self):
        """Softmax output probabilities should sum to 1.0."""
        try:
            import tensorflow as tf
            from models.lstm_model import build_lstm_model, MAX_SEQUENCE_LENGTH
            model   = build_lstm_model()
            dummy_X = np.zeros((2, MAX_SEQUENCE_LENGTH), dtype=np.int32)
            output  = model.predict(dummy_X, verbose=0)
            sums    = output.sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-6)
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_predicted_class_within_range(self):
        """Predicted class should be 0, 1, or 2 only."""
        try:
            import tensorflow as tf
            from models.lstm_model import build_lstm_model, MAX_SEQUENCE_LENGTH
            model   = build_lstm_model()
            dummy_X = np.zeros((5, MAX_SEQUENCE_LENGTH), dtype=np.int32)
            output  = model.predict(dummy_X, verbose=0)
            preds   = np.argmax(output, axis=1)
            assert all(0 <= p <= 2 for p in preds)
        except ImportError:
            pytest.skip("TensorFlow not installed")


# ── DASS-21 Scoring Tests (from app) ─────────────────────────
class TestDASS21AppScoring:

    def compute_scores(self, answers):
        """Helper — compute scores the same way app does."""
        depression_items = [3, 5, 10, 13, 16, 17, 21]
        anxiety_items    = [2, 4,  7,  9, 15, 19, 20]
        stress_items     = [1, 6,  8, 11, 12, 14, 18]
        return {
            "depression": sum(answers.get(f"q{i}", 0) for i in depression_items) * 2,
            "anxiety"   : sum(answers.get(f"q{i}", 0) for i in anxiety_items)    * 2,
            "stress"    : sum(answers.get(f"q{i}", 0) for i in stress_items)     * 2,
        }

    def test_all_zeros_gives_zero_scores(self):
        answers = {f"q{i}": 0 for i in range(1, 22)}
        scores  = self.compute_scores(answers)
        assert scores["depression"] == 0
        assert scores["anxiety"]    == 0
        assert scores["stress"]     == 0

    def test_all_threes_gives_max_scores(self):
        answers = {f"q{i}": 3 for i in range(1, 22)}
        scores  = self.compute_scores(answers)
        assert scores["depression"] == 42
        assert scores["anxiety"]    == 42
        assert scores["stress"]     == 42

    def test_scores_are_always_positive(self):
        answers = {f"q{i}": 1 for i in range(1, 22)}
        scores  = self.compute_scores(answers)
        assert all(v >= 0 for v in scores.values())


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
