# ============================================================
#  MindPulse — Signal Preprocessing Unit Tests
#  Author  : Hanzla (NLP Specialist & QA Lead)
#  File    : tests/test_signal_preprocessing.py
#  Run     : pytest tests/test_signal_preprocessing.py -v
# ============================================================

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.signal_preprocessing import (
    SignalPreprocessor,
    WESAD_SAMPLING_RATES,
    WINDOW_SECONDS,
)


@pytest.fixture
def sp():
    return SignalPreprocessor()


class TestBandpassFilter:

    def test_output_same_length_as_input(self, sp):
        raw = np.random.randn(7000)
        filtered = sp.apply_bandpass_filter(raw, 700, 0.5, 45.0)
        assert len(filtered) == len(raw)

    def test_output_is_numpy_array(self, sp):
        raw = np.random.randn(700)
        filtered = sp.apply_bandpass_filter(raw, 700, 0.5, 45.0)
        assert isinstance(filtered, np.ndarray)

    def test_output_has_no_nan(self, sp):
        raw = np.random.randn(700)
        filtered = sp.apply_bandpass_filter(raw, 700, 0.5, 45.0)
        assert not np.isnan(filtered).any()


class TestSegmentation:

    def test_correct_window_count(self, sp):
        # 2 minutes at 700Hz, 60s windows, 30s overlap → 3 windows
        duration = 120
        rate = 700
        fake_ecg = np.random.randn(duration * rate)
        windows = sp.segment_signal(fake_ecg, rate)
        assert len(windows) == 3

    def test_each_window_correct_length(self, sp):
        rate = 700
        fake_ecg = np.random.randn(300 * rate)   # 5 minutes
        windows = sp.segment_signal(fake_ecg, rate)
        expected_size = WINDOW_SECONDS * rate
        for w in windows:
            assert len(w) == expected_size

    def test_short_signal_gives_no_windows(self, sp):
        # Signal shorter than one window → empty list
        rate = 700
        short_signal = np.random.randn(10 * rate)   # only 10 seconds
        windows = sp.segment_signal(short_signal, rate)
        assert len(windows) == 0


class TestTimeFeatures:

    def test_returns_all_expected_keys(self, sp):
        window = np.random.randn(700)
        features = sp.extract_time_features(window, prefix="ecg_")
        expected = ["ecg_mean", "ecg_std", "ecg_min", "ecg_max",
                    "ecg_range", "ecg_rms", "ecg_peaks"]
        for key in expected:
            assert key in features

    def test_values_are_finite(self, sp):
        window = np.random.randn(700)
        features = sp.extract_time_features(window, prefix="test_")
        for key, val in features.items():
            assert np.isfinite(val), f"{key} is not finite"

    def test_range_equals_max_minus_min(self, sp):
        window = np.array([1.0, 2.0, 3.0, 10.0, -5.0])
        features = sp.extract_time_features(window, prefix="t_")
        assert features["t_range"] == pytest.approx(15.0)


class TestFrequencyFeatures:

    def test_returns_expected_keys(self, sp):
        window = np.random.randn(700)
        features = sp.extract_freq_features(window, 700, prefix="ecg_")
        assert "ecg_lf_power" in features
        assert "ecg_hf_power" in features
        assert "ecg_lf_hf_ratio" in features

    def test_no_division_by_zero_error(self, sp):
        # Flat signal — should not crash even with zero HF power
        window = np.zeros(700)
        features = sp.extract_freq_features(window, 700, prefix="flat_")
        assert features["flat_lf_hf_ratio"] == 0.0


class TestHRVFeatures:

    def test_returns_expected_keys(self, sp):
        window = np.random.randn(700)
        features = sp.extract_hrv_features(window, 700)
        assert "hrv_mean_rr" in features
        assert "hrv_rmssd" in features
        assert "hrv_sdnn" in features

    def test_too_few_peaks_returns_zeros(self, sp):
        # Flat signal has no real peaks
        flat_window = np.zeros(700)
        features = sp.extract_hrv_features(flat_window, 700)
        assert features["hrv_mean_rr"] == 0.0


class TestFeatureVector:

    def test_consistent_length_across_windows(self, sp):
        ecg  = np.random.randn(42000)
        eda  = np.random.randn(240)
        temp = np.random.randn(240)

        v1 = sp.build_feature_vector(ecg, eda, temp)
        v2 = sp.build_feature_vector(
            np.random.randn(42000), np.random.randn(240), np.random.randn(240)
        )
        assert len(v1) == len(v2)

    def test_returns_numpy_array(self, sp):
        vec = sp.build_feature_vector(
            np.random.randn(42000), np.random.randn(240), np.random.randn(240)
        )
        assert isinstance(vec, np.ndarray)


class TestNormalisation:

    def test_output_shape_matches_input(self, sp):
        matrix = np.random.randn(100, 27)
        norm, means, stds = sp.z_score_normalise(matrix)
        assert norm.shape == matrix.shape

    def test_mean_near_zero_after_normalisation(self, sp):
        matrix = np.random.randn(200, 10) * 100 + 50
        norm, _, _ = sp.z_score_normalise(matrix)
        col_means = np.abs(np.mean(norm, axis=0))
        assert np.all(col_means < 1e-8)

    def test_handles_zero_variance_column(self, sp):
        # One column is constant — should not divide by zero
        matrix = np.random.randn(50, 5)
        matrix[:, 0] = 7.0   # constant column
        norm, means, stds = sp.z_score_normalise(matrix)
        assert not np.isnan(norm).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
