"""
p1_dashboard.py — Dashboard / Home page.
Shows system overview, model status, active dataset info, and quick-start guide.
"""

import streamlit as st
from core.database  import get_datasets, get_latest_training_run
from core.ml_engine import model_is_trained
from core.auth      import current_user
from pathlib        import Path
from datetime       import datetime


def show():
    user = current_user()
    role = user.get("role", "student")

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:2rem 0 1.5rem 0;'>
        <div style='font-size:2.2rem; font-weight:800;
             background:linear-gradient(90deg,#60a5fa,#818cf8,#c084fc);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             background-clip:text;'>
            Welcome to AcademiQ Predict
        </div>
        <div style='color:#7c8db5; font-size:1rem; margin-top:0.3rem;'>
            AI-powered student performance prediction using Decision Tree (J48/C4.5)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status Cards ─────────────────────────────────────────────────────────
    trained  = model_is_trained()
    datasets = get_datasets()
    last_run = get_latest_training_run()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🧠 Model Status",
                  "Trained ✅" if trained else "Not Trained",
                  delta="Ready" if trained else "Needs Training",
                  delta_color="normal" if trained else "inverse")

    with col2:
        st.metric("📂 Datasets Loaded", len(datasets),
                  delta="Available" if datasets else "Upload a dataset")

    with col3:
        if last_run:
            st.metric("🎯 Last Accuracy",
                      f"{last_run['accuracy']:.1f}%",
                      delta=f"F1: {last_run['f1_score']:.1f}%")
        else:
            st.metric("🎯 Last Accuracy", "—", delta="No runs yet")

    with col4:
        if last_run:
            ts = last_run["trained_at"][:16].replace("T", " ")
            st.metric("🕐 Last Trained", ts)
        else:
            st.metric("🕐 Last Trained", "—")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Two-column layout ──────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        # ── Feature Overview ─────────────────────────────────────────────
        st.markdown("#### 🗂️ System Features")

        features = [
            ("📂", "Dataset Upload",    "Load CSV or Excel datasets and validate schema automatically."),
            ("🧠", "Model Training",    "Train a J48-equivalent Decision Tree with entropy criterion."),
            ("📈", "Evaluation Metrics","Accuracy, Precision, Recall, F1-Score, Confusion Matrix."),
            ("🌳", "Tree Visualization","Graphical tree viewer + text-based split rules export."),
            ("🔍", "Single Prediction", "Enter one student's data and get an instant classification."),
            ("⚡", "Batch Prediction",  "Upload thousands of unlabelled rows and download results."),
        ]

        for icon, title, desc in features:
            st.markdown(f"""
            <div style='display:flex; align-items:flex-start; gap:1rem;
                        background:#1e2538; border:1px solid #2d3555;
                        border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:0.6rem;'>
                <div style='font-size:1.4rem; flex-shrink:0;'>{icon}</div>
                <div>
                    <div style='font-weight:600; color:#e2e8f0;'>{title}</div>
                    <div style='font-size:0.82rem; color:#7c8db5; margin-top:0.1rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        # ── Quick Start ──────────────────────────────────────────────────
        st.markdown("#### 🚀 Quick Start Guide")

        if role == "admin":
            steps = [
                ("1", "Upload a dataset", "Use the **Dataset Upload** page — or the system auto-loaded `mock_data.csv`."),
                ("2", "Train the model",  "Go to **Model Training** and click *Train Model*."),
                ("3", "Inspect results",  "Review accuracy metrics, confusion matrix, and tree visualization."),
                ("4", "Make predictions","Use **Single Prediction** for one student, or **Batch Prediction** for many."),
                ("5", "Export results",   "Download batch predictions as a CSV file."),
            ]
        else:
            steps = [
                ("1", "Enter student data", "Fill in the form on the **Single Prediction** page."),
                ("2", "Submit",              "Click *Predict Performance* to see your result."),
                ("3", "Interpret result",    "Your classification is colour-coded for clarity."),
            ]

        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; align-items:flex-start; gap:0.9rem;
                        margin-bottom:0.8rem;'>
                <div style='background:linear-gradient(135deg,#3b82f6,#6366f1);
                            color:#fff; font-weight:700; border-radius:50%;
                            width:28px; height:28px; display:flex;
                            align-items:center; justify-content:center;
                            font-size:0.8rem; flex-shrink:0;'>{num}</div>
                <div>
                    <div style='font-weight:600; color:#e2e8f0; font-size:0.9rem;'>{title}</div>
                    <div style='font-size:0.78rem; color:#7c8db5;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Data Schema summary ───────────────────────────────────────────
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 📋 Input Schema")
        schema_data = {
            "Feature":        ["Attendance", "Assignment Score", "Test Score",
                               "Study Hours", "Class Participation", "Previous GPA"],
            "Range":          ["0–100%", "0–100", "0–100",
                               "0–168 hrs", "1–5 (int)", "0.0–4.0"],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(schema_data), width='stretch', hide_index=True)

    # ── Performance Classes Reference ──────────────────────────────────────
    st.markdown("<br/>#### 🏷️ Performance Classes", unsafe_allow_html=False)

    class_info = [
        ("Excellent", "#22c55e", "#14532d", "High across all metrics"),
        ("Good",      "#3b82f6", "#1e3a5f", "Above-average performance"),
        ("Average",   "#f59e0b", "#422006", "Meets basic expectations"),
        ("Poor",      "#f97316", "#431407", "Below satisfactory threshold"),
        ("Fail",      "#ef4444", "#450a0a", "Critical intervention needed"),
    ]

    cols = st.columns(5)
    for col, (label, fg, bg, desc) in zip(cols, class_info):
        with col:
            st.markdown(f"""
            <div style='background:{bg}; border:1px solid {fg}40;
                        border-radius:14px; padding:1rem; text-align:center;'>
                <div style='color:{fg}; font-size:1rem; font-weight:700;'>{label}</div>
                <div style='color:{fg}90; font-size:0.72rem; margin-top:0.3rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Recent datasets ────────────────────────────────────────────────────
    if datasets and role == "admin":
        st.markdown("<br/>#### 📌 Recent Datasets", unsafe_allow_html=False)
        import pandas as pd
        df_ds = pd.DataFrame(datasets)[["filename","row_count","uploaded_at","uploaded_by"]]
        df_ds.columns = ["File Name", "Rows", "Uploaded At", "Uploaded By"]
        st.dataframe(df_ds.head(5), width='stretch', hide_index=True)
