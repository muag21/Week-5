# ============================================================
#  MindPulse — Unit Test Suite
#  Author  : Hanzla (NLP Specialist & QA Lead)
#  File    : tests/test_preprocessing.py
#  Run     : pytest tests/test_preprocessing.py -v
# ============================================================

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.text_preprocessing import (
    TextPreprocessor,
    prepare_smhd_dataset,
    MAX_SEQUENCE_LENGTH,
    MIN_POST_LENGTH,
)
from data.dass_processing import (
    compute_subscale_scores,
    classify_label,
    build_feature_vector,
    normalise_scores,
    THRESHOLDS,
    ANXIETY_ITEMS,
    DEPRESSION_ITEMS,
    STRESS_ITEMS,
)


# ════════════════════════════════════════════════════
#  FIXTURES
# ════════════════════════════════════════════════════

@pytest.fixture
def processor():
    return TextPreprocessor()


@pytest.fixture
def fitted_processor():
    """A TextPreprocessor that already has a vocabulary built."""
    tp = TextPreprocessor()
    texts = [
        "i feel very hopeless and sad every single day cannot get out of bed",
        "my anxiety is terrible today heart racing cannot breathe properly at all",
        "had a great day went to park with friends feeling happy and relaxed",
        "i am so stressed about everything work family money nothing is going well",
    ]
    tp.build_vocab(texts)
    return tp


@pytest.fixture
def sample_data():
    texts = [
        "i have been feeling really hopeless and sad for weeks now cannot get out of bed at all",
        "everything makes me so anxious all the time my heart races constantly cannot sleep properly",
        "had a great day at the park with friends feeling good and well rested today",
        "too short",
    ]
    labels = [1, 2, 0, 0]
    return texts, labels


# ════════════════════════════════════════════════════
#  TEXT CLEANING TESTS
# ════════════════════════════════════════════════════

class TestCleanText:

    def test_removes_http_url(self, processor):
        result = processor.clean_text("check http://example.com for help")
        assert "http" not in result
        assert "example" not in result

    def test_removes_www_url(self, processor):
        result = processor.clean_text("visit www.nhs.uk for support")
        assert "www" not in result

    def test_removes_reddit_mention(self, processor):
        result = processor.clean_text("thanks u/helpfuluser for the advice")
        assert "helpfuluser" not in result

    def test_removes_subreddit(self, processor):
        result = processor.clean_text("posted on r/depression yesterday")
        assert "r/depression" not in result

    def test_converts_to_lowercase(self, processor):
        result = processor.clean_text("I Feel VERY Anxious TODAY")
        assert result == result.lower()

    def test_none_returns_empty_string(self, processor):
        assert processor.clean_text(None) == ""

    def test_empty_string_returns_empty(self, processor):
        assert processor.clean_text("") == ""

    def test_meaningful_words_survive(self, processor):
        result = processor.clean_text("I feel depressed and anxious every day")
        assert "depressed" in result
        assert "anxious" in result

    def test_no_double_spaces(self, processor):
        result = processor.clean_text("I   feel   really   bad   today")
        assert "  " not in result

    def test_no_leading_trailing_spaces(self, processor):
        result = processor.clean_text("  i feel bad  ")
        assert result == result.strip()

    def test_html_entities_removed(self, processor):
        result = processor.clean_text("feeling sad &amp; lonely today")
        assert "&amp;" not in result


# ════════════════════════════════════════════════════
#  POST VALIDITY TESTS
# ════════════════════════════════════════════════════

class TestIsValidPost:

    def test_long_post_is_valid(self, processor):
        long_text = "i feel bad today " * 10
        assert processor.is_valid_post(long_text) is True

    def test_short_post_is_invalid(self, processor):
        assert processor.is_valid_post("I feel bad") is False

    def test_empty_string_is_invalid(self, processor):
        assert processor.is_valid_post("") is False

    def test_exactly_minimum_is_valid(self, processor):
        exact = "word " * MIN_POST_LENGTH
        assert processor.is_valid_post(exact) is True


# ════════════════════════════════════════════════════
#  VOCABULARY TESTS
# ════════════════════════════════════════════════════

class TestVocabulary:

    def test_pad_token_is_index_zero(self, fitted_processor):
        assert fitted_processor.word_to_index["<PAD>"] == 0

    def test_unk_token_is_index_one(self, fitted_processor):
        assert fitted_processor.word_to_index["<UNK>"] == 1

    def test_known_word_encodes_correctly(self, fitted_processor):
        encoded = fitted_processor.encode("i feel hopeless")
        assert all(isinstance(i, int) for i in encoded)
        assert encoded[0] != 1   # "i" should be a known word, not UNK

    def test_unknown_word_maps_to_unk(self, fitted_processor):
        encoded = fitted_processor.encode("supercalifragilistic")
        assert encoded[0] == 1   # index 1 = <UNK>

    def test_encode_raises_without_vocab(self, processor):
        with pytest.raises(RuntimeError):
            processor.encode("some text here today and more words")


# ════════════════════════════════════════════════════
#  PADDING TESTS
# ════════════════════════════════════════════════════

class TestPadSequence:

    def test_short_sequence_padded_to_max_length(self, processor):
        padded = processor.pad_sequence([1, 2, 3])
        assert len(padded) == MAX_SEQUENCE_LENGTH

    def test_padding_fills_with_zeros(self, processor):
        padded = processor.pad_sequence([1, 2, 3])
        assert padded[3] == 0
        assert padded[-1] == 0

    def test_long_sequence_truncated_to_max_length(self, processor):
        long_seq = list(range(MAX_SEQUENCE_LENGTH + 50))
        padded   = processor.pad_sequence(long_seq)
        assert len(padded) == MAX_SEQUENCE_LENGTH

    def test_truncation_keeps_end_of_sequence(self, processor):
        # Put a unique marker at the very end
        long_seq = [0] * MAX_SEQUENCE_LENGTH + [999]
        padded   = processor.pad_sequence(long_seq)
        assert 999 in padded   # end should be kept

    def test_returns_numpy_array(self, processor):
        padded = processor.pad_sequence([1, 2, 3])
        assert isinstance(padded, np.ndarray)


# ════════════════════════════════════════════════════
#  DASS-21 SCORING TESTS
# ════════════════════════════════════════════════════

class TestDASS21Scoring:

    def _all_answers(self, value: int) -> dict:
        return {f"q{i}": value for i in range(1, 22)}

    def test_all_zeros_gives_zero_scores(self):
        scores = compute_subscale_scores(self._all_answers(0))
        assert scores["depression"] == 0
        assert scores["anxiety"]    == 0
        assert scores["stress"]     == 0

    def test_all_threes_gives_max_scores(self):
        scores = compute_subscale_scores(self._all_answers(3))
        assert scores["depression"] == 42
        assert scores["anxiety"]    == 42
        assert scores["stress"]     == 42

    def test_depression_threshold_triggers_label(self):
        answers = self._all_answers(0)
        for q in DEPRESSION_ITEMS:
            answers[f"q{q}"] = 1   # 7 items × 1 × 2 = 14 (exactly at threshold)
        scores = compute_subscale_scores(answers)
        label, _ = classify_label(scores)
        assert label == "depression"

    def test_anxiety_threshold_triggers_label(self):
        answers = self._all_answers(0)
        for q in ANXIETY_ITEMS:
            answers[f"q{q}"] = 1   # 7 × 1 × 2 = 14 → above threshold of 10
        scores = compute_subscale_scores(answers)
        label, _ = classify_label(scores)
        assert label == "anxiety"

    def test_all_zeros_gives_control_label(self):
        scores = compute_subscale_scores(self._all_answers(0))
        label, _ = classify_label(scores)
        assert label == "control"

    def test_normalised_scores_between_zero_and_one(self):
        scores = compute_subscale_scores(self._all_answers(2))
        norm   = normalise_scores(scores)
        for val in norm.values():
            assert 0.0 <= val <= 1.0

    def test_feature_vector_shape(self):
        answers = self._all_answers(1)
        vec = build_feature_vector(answers)
        assert vec.shape == (24,)   # 21 items + 3 subscale scores


# ════════════════════════════════════════════════════
#  FULL PIPELINE TEST
# ════════════════════════════════════════════════════

class TestFullPipeline:

    def test_prepare_smhd_returns_correct_shapes(self, sample_data):
        texts, labels = sample_data
        X, y, proc = prepare_smhd_dataset(texts, labels)
        assert X.ndim == 2
        assert X.shape[1] == MAX_SEQUENCE_LENGTH
        assert len(X) == len(y)

    def test_short_posts_are_filtered(self):
        texts  = ["too short", "also too short",
                  "this is a long enough post with many words to pass the filter check today"]
        labels = [0, 0, 1]
        X, y, _ = prepare_smhd_dataset(texts, labels)
        assert len(X) == 1
        assert y[0] == 1

    def test_processor_encodes_new_text_after_fitting(self, sample_data):
        texts, labels = sample_data
        _, _, proc = prepare_smhd_dataset(texts, labels)
        new_text = "i feel very hopeless and anxious about everything every single day lately"
        result = proc.process(new_text)
        assert result is not None
        assert len(result) == MAX_SEQUENCE_LENGTH


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
