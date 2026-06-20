# ============================================================
#  MindPulse — Physiological Signal Preprocessing
#  Author  : Puja (ML Engineer)
#  File    : data/signal_preprocessing.py
#  Purpose : Process WESAD biosensor data (ECG, EDA, temp)
#            into feature vectors for the Hybrid model
# ============================================================

import numpy as np
from scipy import signal
from typing import List, Dict, Tuple


# ── Configuration ────────────────────────────────────────────
WESAD_SAMPLING_RATES = {
    "ecg"  : 700,    # Hz
    "eda"  : 4,      # Hz
    "temp" : 4,      # Hz
}

WINDOW_SECONDS  = 60     # Each analysis window = 60 seconds
OVERLAP_SECONDS = 30     # Windows overlap by 30 seconds

LABEL_BASELINE = 1
LABEL_STRESS   = 2


class SignalPreprocessor:
    """
    Processes raw physiological signals from the WESAD dataset.

    Pipeline:
        1. apply_bandpass_filter() — remove noise
        2. segment_signal()        — split into windows
        3. extract_time_features() — mean, std, min, max etc.
        4. extract_freq_features() — FFT-based features
        5. extract_hrv_features()  — heart rate variability
        6. build_feature_vector()  — combine everything
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
        Remove noise outside the frequency range of interest.

        Recommended ranges (from WESAD paper):
            ECG : 0.5 Hz to 45.0 Hz
            EDA : 0.05 Hz to 5.0 Hz

        Args:
            signal_data   : 1D array of raw sensor values
            sampling_rate : Samples per second
            low_freq      : Keep frequencies ABOVE this
            high_freq     : Keep frequencies BELOW this
            order         : Filter sharpness (4 = standard)

        Returns:
            Filtered 1D numpy array, same length as input
        """
        nyquist = sampling_rate / 2.0
        low  = low_freq  / nyquist
        high = high_freq / nyquist

        b, a = signal.butter(order, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered

    def segment_signal(
        self,
        signal_data  : np.ndarray,
        sampling_rate: float,
    ) -> List[np.ndarray]:
        """
        Split a long recording into overlapping windows.

        Overlapping windows (30s overlap on 60s windows) ensure
        we don't accidentally cut a stress event in half.

        Args:
            signal_data   : 1D array of sensor values
            sampling_rate : Samples per second

        Returns:
            List of numpy arrays — one per window
        """
        window_size = int(WINDOW_SECONDS  * sampling_rate)
        step_size   = int(OVERLAP_SECONDS * sampling_rate)

        windows = []
        start = 0
        while start + window_size <= len(signal_data):
            windows.append(signal_data[start : start + window_size])
            start += step_size

        return windows

    def extract_time_features(
        self,
        window: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Calculate basic statistical features from one signal window.

        Features: mean, std, min, max, range, rms, peak count.
        Even simple stats like mean/std of EDA can classify
        stress with around 75% accuracy on their own.

        Args:
            window : 1D numpy array (one time window)
            prefix : Label prefix e.g. "ecg_" or "eda_"

        Returns:
            Dictionary of {feature_name: value}
        """
        peak_indices, _ = signal.find_peaks(window, height=0)

        return {
            f"{prefix}mean"  : float(np.mean(window)),
            f"{prefix}std"   : float(np.std(window)),
            f"{prefix}min"   : float(np.min(window)),
            f"{prefix}max"   : float(np.max(window)),
            f"{prefix}range" : float(np.max(window) - np.min(window)),
            f"{prefix}rms"   : float(np.sqrt(np.mean(window ** 2))),
            f"{prefix}peaks" : float(len(peak_indices)),
        }

    def extract_freq_features(
        self,
        window       : np.ndarray,
        sampling_rate: float,
        prefix       : str = "",
    ) -> Dict[str, float]:
        """
        Use FFT to find frequency-domain stress markers.

        During stress, the balance between low-frequency (LF) and
        high-frequency (HF) heart activity shifts. The LF/HF ratio
        is a well-validated stress biomarker in research.

        Bands used:
            LF : 0.04–0.15 Hz
            HF : 0.15–0.40 Hz

        Args:
            window        : 1D numpy array
            sampling_rate : Samples per second
            prefix        : Label prefix

        Returns:
            Dictionary with lf_power, hf_power, lf_hf_ratio
        """
        n = len(window)
        fft_values = np.fft.rfft(window)
        fft_power  = np.abs(fft_values) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)

        lf_power = float(np.sum(fft_power[lf_mask]))
        hf_power = float(np.sum(fft_power[hf_mask]))
        lf_hf_ratio = lf_power / hf_power if hf_power > 1e-6 else 0.0

        return {
            f"{prefix}lf_power"    : lf_power,
            f"{prefix}hf_power"    : hf_power,
            f"{prefix}lf_hf_ratio" : lf_hf_ratio,
        }

    def extract_hrv_features(
        self,
        ecg_window   : np.ndarray,
        sampling_rate: float,
    ) -> Dict[str, float]:
        """
        Calculate Heart Rate Variability (HRV) from an ECG window.

        HIGH variability = relaxed.  LOW variability = stressed.
        We detect R-peaks (heartbeat spikes), measure time between
        them (R-R intervals), then measure how much they vary.

        Args:
            ecg_window    : Filtered ECG signal
            sampling_rate : ECG sampling rate (700 Hz for WESAD)

        Returns:
            Dictionary: hrv_mean_rr, hrv_rmssd, hrv_sdnn
        """
        min_distance = int(0.3 * sampling_rate)   # max ~200 BPM
        r_peaks, _ = signal.find_peaks(
            ecg_window,
            distance=min_distance,
            height=np.mean(ecg_window) + 0.5 * np.std(ecg_window),
        )

        if len(r_peaks) < 3:
            return {"hrv_mean_rr": 0.0, "hrv_rmssd": 0.0, "hrv_sdnn": 0.0}

        rr_intervals_ms = np.diff(r_peaks) / sampling_rate * 1000.0
        successive_diffs = np.diff(rr_intervals_ms)

        return {
            "hrv_mean_rr" : float(np.mean(rr_intervals_ms)),
            "hrv_rmssd"   : float(np.sqrt(np.mean(successive_diffs ** 2))),
            "hrv_sdnn"    : float(np.std(rr_intervals_ms)),
        }

    def build_feature_vector(
        self,
        ecg_window  : np.ndarray,
        eda_window  : np.ndarray,
        temp_window : np.ndarray,
    ) -> np.ndarray:
        """
        Combine all features into one flat array for the Hybrid model.

        Expected output: ~27 features total
            ECG  — 7 time + 3 freq + 3 HRV = 13
            EDA  — 7 time + 3 freq         = 10
            Temp — 4 time-only              = 4

        Args:
            ecg_window  : Filtered ECG window
            eda_window  : Filtered EDA window
            temp_window : Temperature window

        Returns:
            1D numpy array of all features, sorted by key name
        """
        all_features = {}

        all_features.update(self.extract_time_features(ecg_window, "ecg_"))
        all_features.update(self.extract_freq_features(ecg_window, WESAD_SAMPLING_RATES["ecg"], "ecg_"))
        all_features.update(self.extract_hrv_features(ecg_window, WESAD_SAMPLING_RATES["ecg"]))

        all_features.update(self.extract_time_features(eda_window, "eda_"))
        all_features.update(self.extract_freq_features(eda_window, WESAD_SAMPLING_RATES["eda"], "eda_"))

        temp_feats = self.extract_time_features(temp_window, "temp_")
        for key in ["temp_mean", "temp_std", "temp_range", "temp_rms"]:
            all_features[key] = temp_feats.get(key, 0.0)

        return np.array([all_features[k] for k in sorted(all_features.keys())])

    def z_score_normalise(
        self,
        feature_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardise features so each has mean=0, std=1.

        Neural networks train poorly when features have very
        different scales (e.g. ECG power in millions vs temp
        around 36). Z-score fixes this.

        IMPORTANT: compute mean/std on TRAINING data only,
        then reuse the same values for validation/test sets
        to avoid data leakage.

        Args:
            feature_matrix : Shape (N_windows, N_features)

        Returns:
            normalised matrix, means, stds
        """
        means = np.mean(feature_matrix, axis=0)
        stds  = np.std(feature_matrix,  axis=0)
        stds[stds < 1e-8] = 1.0   # avoid divide-by-zero

        normalised = (feature_matrix - means) / stds
        return normalised, means, stds


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Testing SignalPreprocessor")
    print("=" * 50)

    np.random.seed(42)
    sp = SignalPreprocessor()

    duration = 120   # 2 minutes
    ecg_rate = 700
    fake_ecg  = np.random.randn(duration * ecg_rate) * 0.1
    fake_eda  = np.random.randn(duration * WESAD_SAMPLING_RATES["eda"]) * 0.5 + 5
    fake_temp = np.random.randn(duration * WESAD_SAMPLING_RATES["temp"]) * 0.3 + 36.5

    filtered_ecg = sp.apply_bandpass_filter(fake_ecg, ecg_rate, 0.5, 45.0)
    ecg_windows  = sp.segment_signal(filtered_ecg, ecg_rate)
    eda_windows  = sp.segment_signal(fake_eda,  WESAD_SAMPLING_RATES["eda"])
    temp_windows = sp.segment_signal(fake_temp, WESAD_SAMPLING_RATES["temp"])

    print(f"Number of windows  : {len(ecg_windows)}")
    print(f"Window size (ECG)  : {len(ecg_windows[0])} samples")

    feature_vec = sp.build_feature_vector(ecg_windows[0], eda_windows[0], temp_windows[0])
    print(f"Feature vector shape : {feature_vec.shape}")
    print(f"First 5 values       : {feature_vec[:5].round(4)}")

    print("\nAll tests passed!")
