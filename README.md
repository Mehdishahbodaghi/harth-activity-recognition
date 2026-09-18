[README.md](https://github.com/user-attachments/files/32383743/README.md)
# harth-activity-recognition# Wearable Sensor-Based Human Activity Recognition Using Machine Learning

Classifying human physical activity (walking, running, sitting, cycling, stairs, and more) from
raw wearable accelerometer data — comparing classical machine learning (XGBoost, LightGBM)
against deep learning (1D CNN, LSTM) on a real, research-grade sensor dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Results](#results)
- [Key Finding](#key-finding)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Next Steps](#next-steps)
- [Author](#author)

## Project Overview

| | |
|---|---|
| **Task** | Multi-class classification (12 activity classes) |
| **Data** | Real wearable accelerometer recordings, 22 human subjects |
| **Models compared** | XGBoost, LightGBM (classical ML) vs. 1D CNN, LSTM (deep learning, PyTorch) |
| **Evaluation** | Subject-wise train/test split (5 subjects held out entirely for test) |

```
Raw Accelerometer Data
        ↓
Signal Processing
        ↓
Windowing
        ↓
Feature Extraction  ──────►  Classical ML (XGBoost, LightGBM)
        │
        └────────────────►  Deep Learning on raw windows (1D CNN, LSTM)
        ↓
Activity Classification
```

## Dataset

[**HARTH (Human Activity Recognition Trondheim)**](https://archive.ics.uci.edu/dataset/779/harth)
— a real, research-grade dataset (not synthetic):

- **22 subjects**, recorded during free-living daily activity
- **Two 3-axis accelerometers** per subject: lower back + right thigh
- **50 Hz** sampling rate
- **~6.46 million** labeled readings across 12 activity classes: walking, running, shuffling,
  stairs (ascending/descending), standing, sitting, lying, and cycling (sitting/standing,
  active/inactive)

The raw per-subject CSVs (~816MB total) are **not** committed to this repo — see
[How to Run](#how-to-run) for how to get them.

## Pipeline

1. **[EDA](notebooks/01_eda.ipynb)** — raw signal exploration, class distribution, per-subject
   data volume, example waveforms per activity
2. **[Signal processing & windowing](notebooks/02_preprocessing_and_windowing.ipynb)** —
   segments the continuous 50Hz stream into 3-second windows (50% overlap, majority-vote
   labeling with a 90% purity threshold, built per-subject); extracts an 80-dimensional
   hand-crafted feature set (time + frequency domain, per axis, plus cross-axis features) for
   the classical models
3. **[Classical ML](notebooks/03_classical_ml.ipynb)** — trains XGBoost and LightGBM on the
   feature table
4. **[Deep learning](notebooks/04_deep_learning.ipynb)** — trains a 1D CNN and a bidirectional
   LSTM directly on the raw windowed signal (PyTorch)
5. **[Model comparison](notebooks/05_model_comparison.ipynb)** — side-by-side comparison and
   discussion of what the results actually mean

**Methodology note:** all evaluation uses a **subject-wise split** — 5 subjects
(`S008, S017, S021, S022, S029`) are held out entirely for testing and never seen during
training. A random split over windows would leak each person's gait/sensor-placement signature
across train and test and inflate accuracy in a way that would not hold up on a new user; see
notebook 02 for the full reasoning.

## Results

Evaluated on the 5 held-out test subjects:

| Model | Accuracy | Macro F1 |
|---|---|---|
| **XGBoost** | **96.9%** | **0.70** |
| LightGBM | 90.5% | 0.58 |
| LSTM | 91.4% | 0.48 |
| 1D CNN | 88.9% | 0.51 |

*(Macro F1 is the more meaningful metric here — the 12 classes are heavily imbalanced, e.g.
`sitting` alone is ~45% of all readings, so accuracy alone rewards getting the common classes
right while ignoring rare ones like stairs or inactive cycling.)*

## Key Finding

**Classical ML (XGBoost) outperformed both deep learning models on this dataset** — which runs
against the common assumption that deep learning automatically wins. The full discussion is in
[notebook 05](notebooks/05_model_comparison.ipynb), but in short:

- With only 22 subjects (17 for training), the CNN/LSTM had relatively little diversity to learn
  person-invariant movement patterns from raw signal alone, and likely partly learned
  subject-specific quirks instead.
- This shows up directly as a **validation-to-test generalization gap**: the CNN reached >98%
  validation accuracy (on held-out windows from training subjects) but only 88.9% on the truly
  new test subjects — a textbook illustration of why the subject-wise split matters.
- The 80 hand-crafted features gave the classical models a strong, domain-informed head start
  that the deep models had to try to discover from raw signal with comparatively little data.

This isn't presented as "deep learning doesn't work for HAR" — with more subjects, data
augmentation, or a pretrained backbone, the deep models would likely close this gap. It's
presented as what the evidence on *this* dataset, with a properly leakage-free evaluation,
actually shows.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # HARTH subject CSVs go here (not committed - see below)
│   └── processed/        # Windowed arrays, feature table, trained weights, results
├── src/
│   ├── data_loading.py       # Loads & normalizes the raw per-subject CSVs
│   ├── windowing.py          # Sliding-window segmentation (per-subject, majority-vote labels)
│   ├── feature_extraction.py # Hand-crafted time/frequency-domain features (vectorized)
│   ├── splitting.py          # Subject-wise train/test split
│   ├── deep_models.py        # PyTorch 1D CNN and LSTM architectures
│   ├── build_processed_data.py  # One-shot script: raw CSVs -> windows + features
│   ├── train_classical.py    # Trains & saves XGBoost + LightGBM
│   └── train_deep.py         # Trains & saves the 1D CNN + LSTM
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_preprocessing_and_windowing.ipynb
    ├── 03_classical_ml.ipynb
    ├── 04_deep_learning.ipynb
    └── 05_model_comparison.ipynb
```

## How to Run

```bash
git clone <repo-url>
cd harth-activity-recognition
pip install -r requirements.txt
```

**Get the data:** download the HARTH dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/779/harth) (or the
[original Trondheim source](https://github.com/ntnu-ai-lab/harth-ml-experiments)) and place the
per-subject CSV files in `data/raw/`.

**Run the full pipeline:**
```bash
python src/build_processed_data.py   # raw CSVs -> windows + features (~40s)
python src/train_classical.py        # trains XGBoost + LightGBM (~1 min)
python src/train_deep.py             # trains 1D CNN + LSTM (~15 min on CPU)
```

Or open the notebooks in order (`01` → `05`) — each documents and explains every step.
Notebook 04 loads the already-trained deep learning weights by default (since full training
takes ~15 minutes on a single CPU core); set `RETRAIN = True` inside it to train from scratch.

## Next Steps

1. **More subjects / more data** — the highest-leverage next step for closing the deep learning
   gap identified above.
2. **Data augmentation** on the raw signal (jittering, time-warping, magnitude scaling).
3. **A pretrained or self-supervised backbone** for the deep models, rather than training from
   scratch.
4. **Hyperparameter tuning** (all four models used fixed, reasonable-but-untuned settings, to
   keep the comparison apples-to-apples on a limited compute budget).
5. **Hybrid approach** — feed a learned CNN embedding into XGBoost alongside (or instead of) the
   hand-crafted features.

## Author

Mehdi (Mahdi Shahbodaghi) — transitioning into data science / machine learning, with a
background in sports science and photography.
