"""
Trains XGBoost and LightGBM on the hand-crafted feature table
(data/processed/features.csv), using the same subject-wise split as the
deep learning models (see splitting.py), so results are directly comparable.

Usage: python src/train_classical.py
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from splitting import subject_split, DEFAULT_TEST_SUBJECTS

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    feat_df = pd.read_csv(os.path.join(PROC_DIR, "features.csv"))
    train_mask, test_mask = subject_split(feat_df["subject"].values, DEFAULT_TEST_SUBJECTS)

    feature_cols = [c for c in feat_df.columns if c not in ("label", "subject", "activity")]
    X_train, X_test = feat_df.loc[train_mask, feature_cols], feat_df.loc[test_mask, feature_cols]

    le = LabelEncoder()
    y_all = le.fit_transform(feat_df["label"])
    y_train, y_test = y_all[train_mask], y_all[test_mask]

    results = {}

    print("Training XGBoost...")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        n_jobs=-1, random_state=42, eval_metric="mlogloss",
    )
    xgb_model.fit(X_train, y_train)
    print(f"  trained in {time.time()-t0:.1f}s")
    y_pred = xgb_model.predict(X_test)
    acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")
    print(f"  XGBoost -> Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str), output_dict=True, zero_division=0)
    results["xgboost"] = {"accuracy": acc, "macro_f1": f1, "report": report}
    xgb_model.save_model(os.path.join(PROC_DIR, "xgboost_model.json"))

    print("Training LightGBM...")
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        n_jobs=-1, random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    print(f"  trained in {time.time()-t0:.1f}s")
    y_pred = lgb_model.predict(X_test)
    acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")
    print(f"  LightGBM -> Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str), output_dict=True, zero_division=0)
    results["lightgbm"] = {"accuracy": acc, "macro_f1": f1, "report": report}
    lgb_model.booster_.save_model(os.path.join(PROC_DIR, "lightgbm_model.txt"))

    with open(os.path.join(PROC_DIR, "classical_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to data/processed/classical_results.json")


if __name__ == "__main__":
    main()
