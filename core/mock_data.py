"""
mock_data.py — Generates a synthetic seed dataset of 200 student records
with realistic distributions across all 5 performance classes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

CLASSES = ["Excellent", "Good", "Average", "Poor", "Fail"]

# ── Per-class feature distributions (mean, std) ───────────────────────────────
# Format: {class: {feature: (mean, std)}}
_DISTS = {
    "Excellent": {
        "Attendance":         (92, 4),
        "Assignment Score":   (88, 5),
        "Test Score":         (90, 5),
        "Study Hours":        (20, 4),
        "Class Participation": (4.6, 0.4),
        "Previous GPA":       (3.8, 0.2),
    },
    "Good": {
        "Attendance":         (80, 6),
        "Assignment Score":   (74, 6),
        "Test Score":         (76, 6),
        "Study Hours":        (14, 4),
        "Class Participation": (3.7, 0.5),
        "Previous GPA":       (3.1, 0.3),
    },
    "Average": {
        "Attendance":         (68, 7),
        "Assignment Score":   (60, 7),
        "Test Score":         (61, 7),
        "Study Hours":        (9, 3),
        "Class Participation": (2.8, 0.6),
        "Previous GPA":       (2.4, 0.4),
    },
    "Poor": {
        "Attendance":         (54, 8),
        "Assignment Score":   (44, 8),
        "Test Score":         (45, 8),
        "Study Hours":        (5, 2),
        "Class Participation": (2.0, 0.6),
        "Previous GPA":       (1.8, 0.4),
    },
    "Fail": {
        "Attendance":         (38, 10),
        "Assignment Score":   (28, 8),
        "Test Score":         (27, 8),
        "Study Hours":        (2, 1.5),
        "Class Participation": (1.3, 0.4),
        "Previous GPA":       (1.0, 0.4),
    },
}

# Rows per class (total = 200)
_CLASS_COUNTS = {
    "Excellent": 35,
    "Good":      50,
    "Average":   60,
    "Poor":      35,
    "Fail":      20,
}


def generate_mock_dataset(seed: int = 42) -> pd.DataFrame:
    """Return a 200-row DataFrame matching the system schema."""
    rng    = np.random.default_rng(seed)
    frames = []
    student_counter = 1

    for cls, n in _CLASS_COUNTS.items():
        dist = _DISTS[cls]
        rows = {}

        rows["Student ID"] = [f"STU{student_counter + i:04d}" for i in range(n)]
        student_counter += n

        # Continuous features — clamp to valid range
        rows["Attendance"] = np.clip(
            rng.normal(dist["Attendance"][0], dist["Attendance"][1], n), 0, 100
        ).round(1)

        rows["Assignment Score"] = np.clip(
            rng.normal(dist["Assignment Score"][0], dist["Assignment Score"][1], n), 0, 100
        ).round(1)

        rows["Test Score"] = np.clip(
            rng.normal(dist["Test Score"][0], dist["Test Score"][1], n), 0, 100
        ).round(1)

        rows["Study Hours"] = np.clip(
            rng.normal(dist["Study Hours"][0], dist["Study Hours"][1], n), 0, 40
        ).round(1)

        # Ordinal 1-5
        raw_cp = rng.normal(
            dist["Class Participation"][0], dist["Class Participation"][1], n
        )
        rows["Class Participation"] = np.clip(np.round(raw_cp), 1, 5).astype(int)

        # GPA 0.0 – 4.0
        rows["Previous GPA"] = np.clip(
            rng.normal(dist["Previous GPA"][0], dist["Previous GPA"][1], n), 0.0, 4.0
        ).round(2)

        rows["Performance Class"] = [cls] * n

        frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True)
    # Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def save_mock_dataset_if_missing():
    """Write mock_data.csv to data/ folder only if no CSV exists there yet."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    target = data_dir / "mock_data.csv"
    if not target.exists():
        df = generate_mock_dataset()
        df.to_csv(target, index=False)
    return str(target)
