"""
Run once: loads all raw subject CSVs, builds windows, extracts classical-ML
features, and caches everything to data/processed/ so the notebooks don't
need to repeat this (slow-ish, ~30-60s) step every time they're opened.

Outputs:
  data/processed/windows_X.npy       raw windows, shape (n, window_size, 6) - for deep learning
  data/processed/windows_y.npy       integer label per window
  data/processed/windows_subject.npy subject id per window (for subject-wise split)
  data/processed/features.csv        hand-crafted feature table - for XGBoost/LightGBM
                                      (includes label, subject, activity columns)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from data_loading import load_all_subjects, add_activity_names, LABEL_MAP
from windowing import make_windows
from feature_extraction import extract_features_batch

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

WINDOW_SIZE = 150   # 3 seconds at 50Hz
OVERLAP = 0.5        # 50% overlap between consecutive windows
MIN_LABEL_PURITY = 0.9  # drop windows that straddle an activity transition


def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    print("Loading raw subject CSVs...")
    t0 = time.time()
    df = load_all_subjects(RAW_DIR)
    print(f"  Loaded {len(df):,} rows from {df['subject'].nunique()} subjects in {time.time()-t0:.1f}s")

    print("Windowing...")
    t0 = time.time()
    X, y, subjects = make_windows(
        df, window_size=WINDOW_SIZE, overlap=OVERLAP, min_label_purity=MIN_LABEL_PURITY
    )
    print(f"  {X.shape[0]:,} windows of shape {X.shape[1:]} in {time.time()-t0:.1f}s")

    print("Extracting classical-ML features...")
    t0 = time.time()
    feat_df = extract_features_batch(X)
    feat_df["label"] = y
    feat_df["subject"] = subjects
    feat_df["activity"] = feat_df["label"].map(LABEL_MAP)
    print(f"  {feat_df.shape} in {time.time()-t0:.1f}s")

    print("Saving to data/processed/ ...")
    np.save(os.path.join(PROC_DIR, "windows_X.npy"), X)
    np.save(os.path.join(PROC_DIR, "windows_y.npy"), y)
    np.save(os.path.join(PROC_DIR, "windows_subject.npy"), subjects)
    feat_df.to_csv(os.path.join(PROC_DIR, "features.csv"), index=False)

    print("Done.")
    print(f"  windows_X.npy       : {X.nbytes/1e6:.1f} MB")
    print(f"  features.csv        : {feat_df.shape}")


if __name__ == "__main__":
    main()
