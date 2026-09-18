"""
Data loading utilities for the HARTH dataset.

The raw per-subject CSV files have slightly inconsistent columns across
subjects (some include a stray index column, e.g. 'Unnamed: 0' or 'index').
This module normalizes all of that into one consistent schema:

    timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label, subject
"""

import glob
import os
import pandas as pd

SENSOR_COLS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
REQUIRED_COLS = ["timestamp"] + SENSOR_COLS + ["label"]

# HARTH activity label codes -> human-readable activity names
LABEL_MAP = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_ascending",
    5: "stairs_descending",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling_sit",
    14: "cycling_stand",
    130: "cycling_sit_inactive",
    140: "cycling_stand_inactive",
}


def load_subject_csv(path: str) -> pd.DataFrame:
    """Load a single subject CSV and normalize its columns."""
    df = pd.read_csv(path)

    # Drop any stray index-like columns some files ship with
    for junk_col in ("Unnamed: 0", "index"):
        if junk_col in df.columns:
            df = df.drop(columns=[junk_col])

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    df = df[REQUIRED_COLS].copy()

    # Downcast sensor columns to float32 to roughly halve memory use
    for col in SENSOR_COLS:
        df[col] = df[col].astype("float32")
    df["label"] = df["label"].astype("int16")

    subject_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "S006"
    df["subject"] = subject_id

    return df


def load_all_subjects(data_dir: str, pattern: str = "S*.csv") -> pd.DataFrame:
    """Load and concatenate all subject CSVs found under data_dir."""
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} found in {data_dir}")

    frames = [load_subject_csv(p) for p in paths]
    full = pd.concat(frames, ignore_index=True)
    return full


def add_activity_names(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Add an 'activity' column mapping numeric codes to readable names."""
    df = df.copy()
    df["activity"] = df[label_col].map(LABEL_MAP)
    return df
