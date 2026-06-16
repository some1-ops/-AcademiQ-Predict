"""
p3_model_training.py — Model training, evaluation metrics, visualizations.
Admin only.
"""

import streamlit as st
import pandas as pd
import numpy  as np
from pathlib import Path

from core.auth       import require_admin, current_user
from core.ml_engine  import (
    train_model, save_model, load_model, model_is_trained,
    plot_confusion_matrix, plot_decision_tree,
    plot_feature_importance, get_text_rules, CLASS_COLOURS,
)
from core.database   import log_training_run, get_latest_training_run
from core.mock_data  import generate_mock_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_active_df() -> pd.DataFrame | None:
    path = st.session_state.get("active_dataset_path")
    if path and Path(path).exists():
        return pd.read_csv(path)
    # Fallback: use the auto-generated mock_data.csv
    fallback = DATA_DIR / "mock_data.csv"
    if fallback.exists():
        return pd.read_csv(str(fallback))
    return None


def show():
    require_admin()
    user = current_user()

    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0;'>
        <div class='section-header'>🧠 Model Training & Evaluation</div>
        <div class='section-sub'>Train the Decision Tree (J48/C4.5 — Entropy criterion) and inspect results</div>
    </div>
    """, unsafe_allow_html=True)

    df = _load_active_df()

    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first (Dataset Upload page) or "
                   "the system will generate mock data automatically.")
        if st.button("🌱 Generate & Use Mock Data Now"):
            mock = generate_mock_dataset()
            mock_path = DATA_DIR / "mock_data.csv"
            mock.to_csv(str(mock_path), index=False)
            st.session_state["active_dataset_path"] = str(mock_path)
            st.session_state["active_dataset_name"] = "mock_data.csv"
            st.rerun()
        return

    # ── Dataset info bar ──────────────────────────────────────────────────
    active_name = st.session_state.get("active_dataset_name", "mock_data.csv")
    st.markdown(f"""
    <div style='background:#1a2040; border:1px solid #2d3555; border-radius:10px;
                padding:0.7rem 1.2rem; margin-bottom:1.2rem; display:flex;
                align-items:center; gap:1rem;'>
        <span style='color:#60a5fa; font-size:1.2rem;'>📂</span>
        <span style='color:#e2e8f0;'><b>Active Dataset:</b> {active_name}</span>
        <span style='color:#4a5580;'>|</span>
        <span style='color:#7c8db5;'>{len(df):,} rows</span>
        <span style='color:#4a5580;'>|</span>
        <span style='color:#7c8db5;'>80 / 20 split</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Train button ──────────────────────────────────────────────────────
    algorithm_choice = st.radio(
        "Select Prediction Engine:",
        options=["J48 Decision Tree (Legacy)", "Random Forest (Recommended)"],
        horizontal=True
    )
    algo_key = "rf" if "Random Forest" in algorithm_choice else "j48"

    col_btn, col_hint = st.columns([2, 5])
    with col_btn:
        train_clicked = st.button("🚀  Train Model", type="primary", use_container_width=True)

    with col_hint:
        if model_is_trained():
            st.markdown(
                '<span class="badge badge-success">✅ A trained model exists — training again will replace it</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="badge badge-warning">⚠️ No model trained yet</span>',
                unsafe_allow_html=True
            )

    # ── Training logic ────────────────────────────────────────────────────
    if train_clicked:
        with st.spinner(f"Training {algorithm_choice}…"):
            try:
                model, metrics = train_model(df, algorithm=algo_key)
                path = save_model(model)

                log_training_run(
                    dataset_id  = None,
                    accuracy    = metrics["accuracy"],
                    precision_s = metrics["precision"],
                    recall_s    = metrics["recall"],
                    f1_score    = metrics["f1"],
                    train_size  = metrics["train_size"],
                    test_size   = metrics["test_size"],
                    trained_by  = user["username"],
                    model_path  = path,
                )

                st.session_state["trained_model"]   = model
                st.session_state["train_metrics"]   = metrics
                st.success("✅ Model trained and saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                return

    # ── Load cached metrics or last-run ──────────────────────────────────
    metrics = st.session_state.get("train_metrics")
    model   = st.session_state.get("trained_model") or load_model()

    if not model_is_trained() and metrics is None:
        st.info("ℹ️ Click **Train Model** above to begin.")
        return

    if metrics is None:
        # Model exists on disk but metrics not in session — retrain to get them
        st.info("ℹ️ Model found on disk. Click **Train Model** to re-evaluate and display metrics.")
        return

    # ── Metric Cards ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Evaluation Metrics")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🎯 Accuracy",   f"{metrics['accuracy']:.2f}%")
    c2.metric("🔬 Precision",  f"{metrics['precision']:.2f}%")
    c3.metric("📡 Recall",     f"{metrics['recall']:.2f}%")
    c4.metric("⚖️ F1-Score",   f"{metrics['f1']:.2f}%")
    c5.metric("🏋️ Train Set",  f"{metrics['train_size']:,}")
    c6.metric("🧪 Test Set",   f"{metrics['test_size']:,}")

    # ── Tabs: visuals ─────────────────────────────────────────────────────
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌳 Decision Tree",
        "🗃️ Confusion Matrix",
        "📊 Feature Importance",
        "📝 Text Rules",
    ])

    with tab1:
        st.markdown("#### Decision Tree Visualization")
        actual_depth = model.get_depth() if hasattr(model, "get_depth") else (model.estimators_[0].get_depth() if hasattr(model, "estimators_") else 4)
        depth_limit = st.slider("Display depth (levels)", 2, min(8, actual_depth or 4), 4,
                                key="tree_depth_slider")
        with st.spinner("Rendering tree…"):
            fig = plot_decision_tree(model, max_depth_display=depth_limit)
            st.pyplot(fig)

        import matplotlib.pyplot as plt
        plt.close(fig)

        st.markdown(f"""
        <div style='background:#1a2040; border:1px solid #2d3555; border-radius:10px;
                    padding:0.75rem 1.2rem; margin-top:0.5rem;'>
            <b style='color:#60a5fa;'>Tree Stats:</b>
            <span style='color:#e2e8f0; margin-left:1rem;'>Full Depth: {model.get_depth() if hasattr(model, "get_depth") else "N/A"}</span>
            <span style='color:#4a5580; margin: 0 0.5rem;'>|</span>
            <span style='color:#e2e8f0;'>Leaf Nodes: {model.get_n_leaves() if hasattr(model, "get_n_leaves") else "N/A"}</span>
            <span style='color:#4a5580; margin: 0 0.5rem;'>|</span>
            <span style='color:#e2e8f0;'>Criterion: {"Gini (RF)" if hasattr(model, "estimators_") else "Entropy (J48)"}</span>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Confusion Matrix")
        fig_cm = plot_confusion_matrix(metrics["cm"], metrics["cm_labels"])
        st.pyplot(fig_cm)
        import matplotlib.pyplot as plt
        plt.close(fig_cm)

        # Per-class accuracy table
        st.markdown("##### Per-Class Report")
        try:
            from sklearn.metrics import classification_report
            lines = metrics["report"].strip().split("\n")
            rows  = []
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 5 and parts[0] not in ("accuracy","macro","weighted"):
                    rows.append({
                        "Class":     parts[0],
                        "Precision": f"{float(parts[1]):.2%}",
                        "Recall":    f"{float(parts[2]):.2%}",
                        "F1-Score":  f"{float(parts[3]):.2%}",
                        "Support":   parts[4],
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        except Exception:
            st.text(metrics["report"])

    with tab3:
        st.markdown("#### Feature Importance (Information Gain)")
        fig_fi = plot_feature_importance(model)
        st.pyplot(fig_fi)
        import matplotlib.pyplot as plt
        plt.close(fig_fi)

        fi_df = pd.DataFrame({
            "Feature":    metrics["feature_names"],
            "Importance": model.feature_importances_.round(4),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_df["Rank"] = range(1, len(fi_df)+1)
        st.dataframe(fi_df[["Rank","Feature","Importance"]], width='stretch', hide_index=True)

    with tab4:
        st.markdown("#### Text Decision Rules")
        st.markdown(
            "Human-readable split rules extracted from the trained tree. "
            "Each `|---` level represents one decision split."
        )
        rules = get_text_rules(model)
        st.code(rules, language="text")

        st.download_button(
            label="⬇️  Download Rules as .txt",
            data=rules.encode(),
            file_name="decision_rules.txt",
            mime="text/plain",
        )
