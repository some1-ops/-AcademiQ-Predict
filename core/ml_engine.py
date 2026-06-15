"""
ml_engine.py — Decision Tree training, evaluation, persistence, and prediction.
Uses entropy criterion (≡ J48/C4.5 Information Gain) via scikit-learn.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder

from core.data_validator import FEATURE_COLUMNS, TARGET_COLUMN

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = DATA_DIR / "model.pkl"

CLASS_ORDER = ["Excellent", "Good", "Average", "Poor", "Fail"]

# ── Colour mapping for classes ────────────────────────────────────────────────
CLASS_COLOURS = {
    "Excellent": "#22c55e",
    "Good":      "#3b82f6",
    "Average":   "#f59e0b",
    "Poor":      "#f97316",
    "Fail":      "#ef4444",
}


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Train a Decision Tree (entropy, J48-equivalent) on the provided DataFrame.

    Returns:
        (fitted_model, metrics_dict)
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
        max_depth=None,        # unconstrained (mirrors WEKA J48 default)
        min_samples_split=2,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Determine labels that actually exist in the test/pred sets
    present = sorted(set(np.concatenate([y_test, y_pred])),
                     key=lambda c: CLASS_ORDER.index(c) if c in CLASS_ORDER else 99)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, average="macro",
                                           zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, average="macro",
                                        zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred, average="macro",
                                    zero_division=0) * 100, 2),
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "cm":         confusion_matrix(y_test, y_pred, labels=present),
        "cm_labels":  present,
        "report":     classification_report(y_test, y_pred, zero_division=0),
        "feature_names": FEATURE_COLUMNS,
        "class_names":   list(model.classes_),
    }

    return model, metrics


# ── Persistence ───────────────────────────────────────────────────────────────
def save_model(model: DecisionTreeClassifier) -> str:
    DATA_DIR.mkdir(exist_ok=True)
    joblib.dump(model, str(MODEL_PATH))
    return str(MODEL_PATH)


def load_model() -> Optional[DecisionTreeClassifier]:
    if MODEL_PATH.exists():
        try:
            return joblib.load(str(MODEL_PATH))
        except Exception:
            return None
    return None


def model_is_trained() -> bool:
    return MODEL_PATH.exists()


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_single(model: DecisionTreeClassifier, feature_values: list) -> str:
    """Predict performance class for a single student feature vector."""
    arr = np.array(feature_values, dtype=float, ndmin=2).reshape(1, -1)
    return model.predict(arr)[0]


def predict_batch(model: DecisionTreeClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """Predict for all rows in df. Adds 'Predicted_Performance' column."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    df = df.copy()
    df["Predicted_Performance"] = model.predict(X)
    return df


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, labels: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="#2d3555",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", color="#a3b3d4", fontsize=11)
    ax.set_ylabel("True Label",      color="#a3b3d4", fontsize=11)
    ax.set_title("Confusion Matrix",  color="#e2e8f0", fontsize=13, pad=12)
    ax.tick_params(colors="#a3b3d4")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="#a3b3d4")
    plt.setp(ax.get_yticklabels(), rotation=0,  color="#a3b3d4")

    # Style colourbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="#a3b3d4")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#a3b3d4")

    fig.tight_layout()
    return fig


def plot_decision_tree(model: DecisionTreeClassifier,
                       max_depth_display: int = 4) -> plt.Figure:
    """Render the decision tree. Limits visual depth for readability."""
    n_classes = len(model.classes_)
    fig_w = max(20, n_classes * 5)
    fig_h = max(10, min(max_depth_display * 3, 20))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#1a1f2e")
    ax.set_facecolor("#1a1f2e")

    plot_tree(
        model,
        max_depth=max_depth_display,
        feature_names=FEATURE_COLUMNS,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        ax=ax,
        fontsize=8,
    )

    ax.set_title(
        f"Decision Tree — top {max_depth_display} levels  "
        f"(full depth: {model.get_depth()})",
        color="#e2e8f0", fontsize=12, pad=8
    )
    fig.tight_layout()
    return fig


def get_text_rules(model: DecisionTreeClassifier) -> str:
    return export_text(model, feature_names=FEATURE_COLUMNS, max_depth=10)


def plot_feature_importance(model: DecisionTreeClassifier) -> plt.Figure:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLUMNS[i] for i in indices]
    sorted_vals     = importances[indices]

    colours = [
        "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7",
        "#ec4899", "#f43f5e",
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")

    bars = ax.barh(sorted_features[::-1], sorted_vals[::-1],
                   color=colours[:len(sorted_features)], edgecolor="none", height=0.6)
    ax.set_xlabel("Importance (Information Gain)", color="#a3b3d4", fontsize=10)
    ax.set_title("Feature Importance", color="#e2e8f0", fontsize=12, pad=10)
    ax.tick_params(colors="#a3b3d4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3555")
    ax.spines["bottom"].set_color("#2d3555")
    ax.xaxis.label.set_color("#a3b3d4")

    for bar, val in zip(bars, sorted_vals[::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#e2e8f0", fontsize=8)

    fig.tight_layout()
    return fig
