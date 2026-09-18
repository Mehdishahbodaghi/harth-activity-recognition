"""
Sliding-window segmentation of the HARTH accelerometer streams.

Windows are built PER SUBJECT (never across a subject boundary) so a window
never mixes signal from two different people. Each window gets a single
label, taken as the majority label within it; windows where the majority
label doesn't dominate strongly enough (i.e. windows straddling an activity
transition) are dropped, since a "sitting/walking" boundary window is not a
clean example of either class.
"""

import numpy as np
import pandas as pd

from data_loading import SENSOR_COLS


def _windows_for_subject(
    df_subj: pd.DataFrame,
    window_size: int,
    step: int,
    min_label_purity: float,
):
    values = df_subj[SENSOR_COLS].to_numpy(dtype="float32")  # (n_samples, 6)
    labels = df_subj["label"].to_numpy()
    n_samples = len(df_subj)

    starts = range(0, n_samples - window_size + 1, step)

    windows = []
    window_labels = []
    for start in starts:
        end = start + window_size
        seg_labels = labels[start:end]
        vals, counts = np.unique(seg_labels, return_counts=True)
        majority_idx = counts.argmax()
        purity = counts[majority_idx] / window_size
        if purity < min_label_purity:
            continue  # transition window - discard
        windows.append(values[start:end])
        window_labels.append(vals[majority_idx])

    if not windows:
        return np.empty((0, window_size, len(SENSOR_COLS)), dtype="float32"), np.empty((0,), dtype="int16")

    return np.stack(windows).astype("float32"), np.array(window_labels, dtype="int16")


def make_windows(
    df: pd.DataFrame,
    window_size: int = 150,
    overlap: float = 0.5,
    min_label_purity: float = 0.9,
):
    """
    Segment df (must contain 'subject', 'label', and SENSOR_COLS) into
    fixed-size windows, grouped per subject.

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size, 6)
        Raw sensor windows, channel order = SENSOR_COLS.
    y : np.ndarray, shape (n_windows,)
        Majority activity label code per window.
    subjects : np.ndarray, shape (n_windows,)
        Subject id each window belongs to (for subject-wise splitting).
    """
    step = max(1, int(window_size * (1 - overlap)))

    all_X, all_y, all_subj = [], [], []
    for subject_id, df_subj in df.groupby("subject", sort=True):
        df_subj = df_subj.reset_index(drop=True)
        X, y = _windows_for_subject(df_subj, window_size, step, min_label_purity)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_subj.extend([subject_id] * len(X))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.array(all_subj)

    return X, y, subjects
