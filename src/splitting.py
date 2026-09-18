"""
Subject-wise train/test split.

Standard random splitting would put windows from the same subject in both
train and test, which leaks subject-specific gait/movement signatures across
the split and inflates accuracy in a way that will NOT hold up on a new
person. Splitting by subject (holding entire subjects out for test) is the
scientifically honest way to evaluate a HAR model.
"""

import numpy as np

# Fixed test subjects (~23% of the 22 subjects) chosen once via a fixed seed
# below, then hardcoded here so every notebook / script uses the exact same
# split and results are directly comparable across models.
DEFAULT_TEST_SUBJECTS = ["S008", "S017", "S021", "S022", "S029"]


def subject_split(subjects: np.ndarray, test_subjects=None):
    """
    Returns boolean masks (train_mask, test_mask) over `subjects`.
    """
    if test_subjects is None:
        test_subjects = DEFAULT_TEST_SUBJECTS
    test_mask = np.isin(subjects, test_subjects)
    train_mask = ~test_mask
    return train_mask, test_mask


def choose_random_test_subjects(all_subjects, n_test=5, seed=42):
    """Utility used once to originally pick DEFAULT_TEST_SUBJECTS."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(sorted(set(all_subjects)), size=n_test, replace=False).tolist())
