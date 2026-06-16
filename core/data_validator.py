"""
data_validator.py — Validates that an uploaded DataFrame conforms to the
system's required column schema and value ranges.
"""

import pandas as pd
from typing import Tuple, List

# ── Required schema ───────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Student ID",
    "Attendance",
    "Assignment Score",
    "Test Score",
    "Study Hours",
    "Class Participation",
    "Previous GPA",
    "Academic_Momentum",
    "Performance Class",
]

FEATURE_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in ("Student ID", "Performance Class")]

TARGET_COLUMN   = "Performance Class"
VALID_CLASSES   = {"Excellent", "Good", "Average", "Poor", "Fail"}

RANGE_RULES = {
    "Attendance":          (0,   100),
    "Assignment Score":    (0,   100),
    "Test Score":          (0,   100),
    "Study Hours":         (0,   168),
    "Class Participation": (1,   5),
    "Previous GPA":        (0.0, 4.0),
    "Academic_Momentum":   (-5.0, 5.0),
}


# ── Public API ────────────────────────────────────────────────────────────────
def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame against the required schema.

    Returns:
        (is_valid: bool, errors: list[str])
    """
    errors: List[str] = []

    # 1. Column presence
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return False, errors  # Can't proceed without the columns

    # 2. Empty dataset
    if len(df) == 0:
        errors.append("The dataset is empty (0 rows).")
        return False, errors

    # 3. Numeric type coercion check
    for col, (lo, hi) in RANGE_RULES.items():
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            n_null = numeric.isna().sum()
            if n_null > 0:
                errors.append(
                    f"Column '{col}' has {n_null} non-numeric value(s). They will be dropped."
                )
        except Exception as e:
            errors.append(f"Column '{col}' could not be parsed: {e}")

    # 4. Target class values
    if TARGET_COLUMN in df.columns:
        bad = set(df[TARGET_COLUMN].dropna().unique()) - VALID_CLASSES
        if bad:
            errors.append(
                f"Column '{TARGET_COLUMN}' contains unrecognised class(es): {bad}. "
                f"Valid values are: {VALID_CLASSES}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce types, drop rows with missing critical values, and
    clip numeric columns to their valid ranges.
    Returns a clean copy.
    """
    df = df.copy()

    # Coerce numerics
    for col in RANGE_RULES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing any feature or target
    df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN], inplace=True)

    # Clip ranges
    for col, (lo, hi) in RANGE_RULES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Ensure Class Participation is integer
    if "Class Participation" in df.columns:
        df["Class Participation"] = df["Class Participation"].round().astype(int)

    # Normalise target labels (strip whitespace)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip().str.title()

    return df


def validate_prediction_input(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate an unlabelled CSV for batch prediction (no target column required)."""
    errors: List[str] = []
    needed = [c for c in REQUIRED_COLUMNS if c != TARGET_COLUMN]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns for prediction: {missing}")
    return len(errors) == 0, errors
