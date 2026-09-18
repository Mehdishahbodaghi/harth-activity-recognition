"""
Hand-crafted statistical + frequency-domain feature extraction for
accelerometer windows. These features feed the classical ML models
(XGBoost, LightGBM), which need a fixed-length feature vector per window,
unlike the deep learning models which consume the raw window directly.

Feature families (per axis, 6 axes: back_x/y/z, thigh_x/y/z):
  - Time domain: mean, std, min, max, median, IQR, skewness, kurtosis,
    root mean square (RMS), zero-crossing rate
  - Frequency domain: dominant frequency, spectral energy (via FFT)
Cross-axis:
  - Signal Magnitude Area (SMA) for back and thigh sensors separately
  - Correlation between axis pairs on the same sensor (back_x/back_y, etc.)

Fully vectorized across all windows at once (no per-window Python loop),
which is what makes this fast enough for ~80k windows.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from data_loading import SENSOR_COLS

SAMPLING_RATE_HZ = 50


def extract_features_batch(X: np.ndarray) -> pd.DataFrame:
    """
    X: array of shape (n_windows, window_size, 6), columns = SENSOR_COLS order.
    Returns a DataFrame with one row per window, fully vectorized.
    """
    n_windows, window_size, n_axes = X.shape
    feats = {}

    for i, axis_name in enumerate(SENSOR_COLS):
        sig = X[:, :, i]  # shape (n_windows, window_size)
        centered = sig - sig.mean(axis=1, keepdims=True)

        feats[f"{axis_name}_mean"] = sig.mean(axis=1)
        feats[f"{axis_name}_std"] = sig.std(axis=1)
        feats[f"{axis_name}_min"] = sig.min(axis=1)
        feats[f"{axis_name}_max"] = sig.max(axis=1)
        feats[f"{axis_name}_median"] = np.median(sig, axis=1)
        feats[f"{axis_name}_iqr"] = np.percentile(sig, 75, axis=1) - np.percentile(sig, 25, axis=1)
        feats[f"{axis_name}_skew"] = skew(sig, axis=1)
        feats[f"{axis_name}_kurtosis"] = kurtosis(sig, axis=1)
        feats[f"{axis_name}_rms"] = np.sqrt(np.mean(sig ** 2, axis=1))

        signs = centered >= 0
        crossings = np.sum(signs[:, :-1] != signs[:, 1:], axis=1)
        feats[f"{axis_name}_zcr"] = crossings / window_size

        fft_vals = np.abs(np.fft.rfft(centered, axis=1))
        freqs = np.fft.rfftfreq(window_size, d=1 / SAMPLING_RATE_HZ)
        fft_no_dc = fft_vals[:, 1:]
        dominant_idx = np.argmax(fft_no_dc, axis=1) + 1
        feats[f"{axis_name}_dominant_freq"] = freqs[dominant_idx]
        feats[f"{axis_name}_spectral_energy"] = np.sum(fft_vals ** 2, axis=1) / window_size

    back = X[:, :, 0:3]
    thigh = X[:, :, 3:6]
    feats["back_sma"] = np.mean(np.sum(np.abs(back), axis=2), axis=1)
    feats["thigh_sma"] = np.mean(np.sum(np.abs(thigh), axis=2), axis=1)

    def batch_corr(a, b):
        a_c = a - a.mean(axis=1, keepdims=True)
        b_c = b - b.mean(axis=1, keepdims=True)
        num = np.sum(a_c * b_c, axis=1)
        denom = np.sqrt(np.sum(a_c ** 2, axis=1) * np.sum(b_c ** 2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, num / denom, 0.0)
        return corr

    for sensor, idx in (("back", (0, 1, 2)), ("thigh", (3, 4, 5))):
        pairs = [(idx[0], idx[1], "xy"), (idx[0], idx[2], "xz"), (idx[1], idx[2], "yz")]
        for a_i, b_i, name in pairs:
            feats[f"{sensor}_corr_{name}"] = batch_corr(X[:, :, a_i], X[:, :, b_i])

    feat_df = pd.DataFrame(feats)

    # A handful of windows are (near-)constant (e.g. inactive cycling), which
    # makes skew/kurtosis mathematically undefined (0/0) -> NaN. A flat
    # signal has no skew or kurtosis, so 0.0 is the semantically correct fill.
    feat_df = feat_df.fillna(0.0)

    return feat_df
