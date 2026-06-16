"""
p5_batch_prediction.py — Batch prediction: upload unlabelled CSV,
predict all rows, preview results, and export to CSV.
Admin only.
"""

import io
import streamlit as st
import pandas    as pd
from pathlib import Path

from core.auth           import require_admin, current_user
from core.ml_engine      import load_model, predict_batch, model_is_trained, CLASS_COLOURS
from core.data_validator import validate_prediction_input, preprocess_dataset, FEATURE_COLS


def show():
    require_admin()
    user = current_user()

    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0;'>
        <div class='section-header'>⚡ Batch Prediction</div>
        <div class='section-sub'>Upload an unlabelled dataset and generate predictions for all students at once</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_is_trained():
        st.warning("⚠️ No trained model found. Please train a model on the **Model Training** page first.")
        return

    model = load_model()
    if model is None:
        st.error("❌ Failed to load model. Please retrain.")
        return

    # ── Instructions card ──────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#1a2040; border:1px solid #2d3555; border-radius:12px;
                padding:1rem 1.4rem; margin-bottom:1.4rem;'>
        <b style='color:#60a5fa;'>📋 How Batch Prediction Works:</b>
        <ol style='color:#a3b3d4; margin:0.5rem 0 0 0; font-size:0.88rem;'>
            <li>Upload a CSV containing student data <em>without</em> a <code>Performance Class</code> column.</li>
            <li>The system validates the schema and runs predictions on every row.</li>
            <li>Results include a <code>Predicted_Performance</code> column (J48 class) and an
                <code>Estimated_CGPA</code> column (heuristic, 5.0 scale).</li>
            <li>Download the full results as CSV or Excel.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ──────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload unlabelled student CSV",
        type=["csv", "xlsx"],
        key="batch_uploader",
        help="The file must contain all required feature columns (except Performance Class).",
    )

    if not uploaded:
        # Show template download
        sample_pred = pd.DataFrame({
            "Student ID":          ["STU0101", "STU0102", "STU0103"],
            "Attendance":          [88.0, 55.0, 72.0],
            "Assignment Score":    [80.0, 42.0, 66.0],
            "Test Score":          [85.0, 38.0, 70.0],
            "Study Hours":         [16.0, 4.0, 11.0],
            "Class Participation": [4, 2, 3],
            "Previous GPA":        [3.5, 1.6, 2.8],
        })
        st.download_button(
            label="⬇️  Download Blank Prediction Template",
            data=sample_pred.to_csv(index=False).encode(),
            file_name="batch_prediction_template.csv",
            mime="text/csv",
        )
        return

    # ── Load file ──────────────────────────────────────────────────────────
    with st.spinner("Reading file…"):
        try:
            if uploaded.name.endswith(".xlsx"):
                df_raw = pd.read_excel(uploaded, engine="openpyxl")
            else:
                df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            return

    is_valid, errors = validate_prediction_input(df_raw)
    if not is_valid:
        for err in errors:
            st.error(err)
        return

    st.success(f"✅ File accepted — **{len(df_raw):,}** rows detected.")

    # ── Preprocess (no target column) ─────────────────────────────────────
    # Temporarily add a dummy target column for preprocess pipeline
    df_raw["Performance Class"] = "Unknown"
    try:
        df_clean = preprocess_dataset(df_raw)
    except Exception as e:
        st.error(f"❌ Preprocessing failed: {e}")
        return
    df_clean = df_clean.drop(columns=["Performance Class"], errors="ignore")

    # ── Preview ───────────────────────────────────────────────────────────
    st.markdown("##### 👁️ Input Preview (first 5 rows)")
    st.dataframe(df_clean.head(5), width='stretch', hide_index=True)

    # ── Run predictions ────────────────────────────────────────────────────
    run_btn = st.button("⚡  Run Batch Prediction", type="primary", use_container_width=False)

    if run_btn:
        with st.spinner(f"Predicting for {len(df_clean):,} students…"):
            try:
                result_df = predict_batch(model, df_clean)
                st.session_state["batch_result"] = result_df
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                return
        st.success(f"✅ Completed — {len(result_df):,} predictions generated!")

    # ── Results display ────────────────────────────────────────────────────
    result_df = st.session_state.get("batch_result")

    if result_df is not None and len(result_df) > 0:
        st.markdown("---")
        st.markdown("#### 📊 Prediction Results")

        # ── Summary distribution ─────────────────────────────────────────
        dist = result_df["Predicted_Performance"].value_counts()

        cols = st.columns(min(len(dist), 5))
        for col, (cls, cnt) in zip(cols, dist.items()):
            meta_fg = CLASS_COLOURS.get(cls, "#888")
            with col:
                st.markdown(f"""
                <div style='background:#1e2538; border:1px solid {meta_fg}44;
                            border-radius:12px; padding:0.8rem; text-align:center;'>
                    <div style='color:{meta_fg}; font-size:1.3rem; font-weight:700;'>{cnt}</div>
                    <div style='color:{meta_fg}88; font-size:0.75rem;'>{cls}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── CGPA summary metrics ────────────────────────────────────────────────
        if "Estimated_CGPA" in result_df.columns:
            cgpa_mean = result_df["Estimated_CGPA"].mean()
            cgpa_max  = result_df["Estimated_CGPA"].max()
            cgpa_min  = result_df["Estimated_CGPA"].min()

            c1, c2, c3 = st.columns(3)
            for col_ui, label, val, colour in [
                (c1, "Avg Estimated CGPA", f"{cgpa_mean:.2f}", "#3b82f6"),
                (c2, "Highest CGPA",       f"{cgpa_max:.2f}",  "#22c55e"),
                (c3, "Lowest CGPA",        f"{cgpa_min:.2f}",  "#f97316"),
            ]:
                with col_ui:
                    st.markdown(f"""
                    <div style='background:#1e2538; border:1px solid {colour}44;
                                border-radius:12px; padding:0.8rem 1rem; text-align:center;'>
                        <div style='color:{colour}; font-size:1.4rem; font-weight:700;'>{val}</div>
                        <div style='color:{colour}88; font-size:0.73rem;'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Distribution bar chart ────────────────────────────────────────
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cls_order = [c for c in ["Excellent","Good","Average","Poor","Fail"]
                     if c in dist.index]
        counts    = [dist.get(c, 0) for c in cls_order]
        colours   = [CLASS_COLOURS.get(c, "#888") for c in cls_order]

        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        bars = ax.bar(cls_order, counts, color=colours, edgecolor="none", width=0.55)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", color="#e2e8f0", fontsize=9)
        ax.set_ylabel("Students", color="#a3b3d4")
        ax.tick_params(colors="#a3b3d4")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#2d3555"); ax.spines["bottom"].set_color("#2d3555")
        ax.set_title("Predicted Performance Distribution", color="#e2e8f0", fontsize=11, pad=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Full results table ────────────────────────────────────────────
        st.markdown("##### 📋 Full Results Table")

        # Filter / search
        filter_classes = st.multiselect(
            "Filter by Predicted Class",
            options=["Excellent","Good","Average","Poor","Fail"],
            default=[],
            key="batch_filter",
        )
        display_df = result_df.copy()
        if filter_classes:
            display_df = display_df[display_df["Predicted_Performance"].isin(filter_classes)]

        st.dataframe(display_df, width='stretch', hide_index=True, height=400)

        # ── Download ──────────────────────────────────────────────────────
        st.markdown("---")
        csv_out = result_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️  Download Predictions as CSV",
            data=csv_out,
            file_name="batch_predictions.csv",
            mime="text/csv",
            type="primary",
        )

        # Also offer Excel
        buf = io.BytesIO()
        result_df.to_excel(buf, index=False, engine="openpyxl")
        st.download_button(
            label="⬇️  Download as Excel",
            data=buf.getvalue(),
            file_name="batch_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # ── Early Warning Dashboard ───────────────────────────────────────
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 🚨 Academic Intervention Dashboard")
        
        at_risk_df = result_df[result_df["Predicted_Performance"].isin(["Poor", "Fail"])]
        if len(at_risk_df) > 0:
            with st.container(border=True):
                st.error(f"**Action Required:** {len(at_risk_df)} student(s) flagged as high-risk.")
                display_cols_risk = ["Student ID", "Predicted_Performance", "Estimated_CGPA"]
                st.dataframe(at_risk_df[[c for c in display_cols_risk if c in at_risk_df.columns]], use_container_width=True, hide_index=True)
        else:
            st.success("✅ No students in this batch are currently at risk of failing.")
