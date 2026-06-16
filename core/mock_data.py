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
        "Attendance_Pct":      (92, 4),
        "CA_Score":            (35, 3),
        "Exam_Score":          (55, 3),
        "Study_Hours_Week":    (20, 4),
        "Class_Participation": (4.6, 0.4),
        "Previous_GPA":        (3.8, 0.2),
        "Academic_Momentum":   (0.2, 0.3),
    },
    "Good": {
        "Attendance_Pct":      (80, 6),
        "CA_Score":            (30, 4),
        "Exam_Score":          (46, 5),
        "Study_Hours_Week":    (14, 4),
        "Class_Participation": (3.7, 0.5),
        "Previous_GPA":        (3.1, 0.3),
        "Academic_Momentum":   (0.1, 0.2),
    },
    "Average": {
        "Attendance_Pct":      (68, 7),
        "CA_Score":            (24, 4),
        "Exam_Score":          (37, 5),
        "Study_Hours_Week":    (9, 3),
        "Class_Participation": (2.8, 0.6),
        "Previous_GPA":        (2.4, 0.4),
        "Academic_Momentum":   (-0.1, 0.3),
    },
    "Poor": {
        "Attendance_Pct":      (54, 8),
        "CA_Score":            (18, 4),
        "Exam_Score":          (27, 5),
        "Study_Hours_Week":    (5, 2),
        "Class_Participation": (2.0, 0.6),
        "Previous_GPA":        (1.8, 0.4),
        "Academic_Momentum":   (-0.3, 0.3),
    },
    "Fail": {
        "Attendance_Pct":      (38, 10),
        "CA_Score":            (11, 4),
        "Exam_Score":          (16, 5),
        "Study_Hours_Week":    (2, 1.5),
        "Class_Participation": (1.3, 0.4),
        "Previous_GPA":        (1.0, 0.4),
        "Academic_Momentum":   (-0.5, 0.2),
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
        rows["Attendance_Pct"] = np.clip(
            rng.normal(dist["Attendance_Pct"][0], dist["Attendance_Pct"][1], n), 0, 100
        ).round(1)

        rows["CA_Score"] = np.clip(
            rng.normal(dist["CA_Score"][0], dist["CA_Score"][1], n), 0, 40
        ).round(1)

        rows["Exam_Score"] = np.clip(
            rng.normal(dist["Exam_Score"][0], dist["Exam_Score"][1], n), 0, 60
        ).round(1)
        
        rows["Total_Score"] = rows["CA_Score"] + rows["Exam_Score"]

        rows["Study_Hours_Week"] = np.clip(
            rng.normal(dist["Study_Hours_Week"][0], dist["Study_Hours_Week"][1], n), 0, 40
        ).round(1)

        # Ordinal 1-5
        raw_cp = rng.normal(
            dist["Class_Participation"][0], dist["Class_Participation"][1], n
        )
        rows["Class_Participation"] = np.clip(np.round(raw_cp), 1, 5).astype(int)

        # GPA 0.0 – 5.0
        rows["Previous_GPA"] = np.clip(
            rng.normal(dist["Previous_GPA"][0], dist["Previous_GPA"][1], n), 0.0, 5.0
        ).round(2)

        # Academic Momentum
        rows["Academic_Momentum"] = np.clip(
            rng.normal(dist["Academic_Momentum"][0], dist["Academic_Momentum"][1], n), -5.0, 5.0
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
