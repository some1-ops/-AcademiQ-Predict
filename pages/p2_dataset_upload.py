"""
p2_dataset_upload.py — Dataset upload, validation, preview, and persistence.
Admin only.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

from core.auth           import require_admin, current_user
from core.data_validator import validate_dataset, preprocess_dataset, REQUIRED_COLUMNS
from core.database       import log_dataset
from core.mock_data      import generate_mock_dataset

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"


def show():
    require_admin()
    user = current_user()

    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0;'>
        <div class='section-header'>📂 Dataset Management</div>
        <div class='section-sub'>Upload, validate, and preview your student dataset</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⬆️  Upload Dataset", "📋 Schema Reference", "🌱 Use Mock Data"])

    # ── TAB 1: Upload ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Upload a Student Dataset")
        st.markdown(
            "Accepted formats: **CSV** (`.csv`) or **Excel** (`.xlsx`). "
            "The file must contain all required columns listed in the *Schema Reference* tab."
        )

        uploaded = st.file_uploader(
            label="Drag and drop file here, or click to browse",
            type=["csv", "xlsx"],
            key="dataset_uploader",
        )

        if uploaded:
            with st.spinner("Reading file…"):
                try:
                    if uploaded.name.endswith(".xlsx"):
                        df_raw = pd.read_excel(uploaded, engine="openpyxl")
                    else:
                        df_raw = pd.read_csv(uploaded)
                except Exception as e:
                    st.error(f"❌ Could not read file: {e}")
                    return

            # ── Validation ──────────────────────────────────────────────
            is_valid, errors = validate_dataset(df_raw)

            if errors:
                with st.expander("⚠️ Validation Warnings", expanded=True):
                    for err in errors:
                        st.warning(err)

            if not is_valid:
                st.error("The dataset failed schema validation. Please fix the issues above and re-upload.")
                return

            st.success(f"✅ Validation passed — **{len(df_raw):,}** rows × **{len(df_raw.columns)}** columns")

            # ── Preprocess & Preview ─────────────────────────────────────
            df_clean = preprocess_dataset(df_raw)
            dropped  = len(df_raw) - len(df_clean)

            if dropped > 0:
                st.info(f"ℹ️ {dropped} row(s) with missing/invalid values were removed during preprocessing.")

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1: st.metric("Total Rows",  f"{len(df_clean):,}")
            with col_stat2: st.metric("Columns",     len(df_clean.columns))
            with col_stat3:
                class_counts = df_clean["Performance Class"].value_counts()
                st.metric("Classes Present", len(class_counts))

            st.markdown("##### 👁️ Data Preview (first 10 rows)")
            st.dataframe(df_clean.head(10), width='stretch', hide_index=True)

            # ── Class distribution bar ───────────────────────────────────
            st.markdown("##### 📊 Class Distribution")
            colours_map = {
                "Excellent": "#22c55e",
                "Good":      "#3b82f6",
                "Average":   "#f59e0b",
                "Poor":      "#f97316",
                "Fail":      "#ef4444",
            }
            dist_df = df_clean["Performance Class"].value_counts().reset_index()
            dist_df.columns = ["Class", "Count"]
            dist_df["Colour"] = dist_df["Class"].map(colours_map).fillna("#888")

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#1e2130")
            ax.set_facecolor("#1e2130")
            bars = ax.bar(dist_df["Class"], dist_df["Count"],
                          color=dist_df["Colour"], edgecolor="none", width=0.55)
            for bar, val in zip(bars, dist_df["Count"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(val), ha="center", va="bottom", color="#e2e8f0", fontsize=9)
            ax.set_ylabel("Count", color="#a3b3d4")
            ax.tick_params(colors="#a3b3d4")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#2d3555")
            ax.spines["bottom"].set_color("#2d3555")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Descriptive stats ────────────────────────────────────────
            with st.expander("📐 Descriptive Statistics"):
                st.dataframe(df_clean.describe().round(2), width='stretch')

            # ── Save button ──────────────────────────────────────────────
            st.markdown("---")
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                if st.button("💾  Save & Activate Dataset", type="primary", use_container_width=True):
                    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                    save_path = UPLOADS_DIR / uploaded.name
                    df_clean.to_csv(str(save_path), index=False)

                    log_dataset(
                        filename=uploaded.name,
                        row_count=len(df_clean),
                        col_count=len(df_clean.columns),
                        file_path=str(save_path),
                        uploaded_by=user["username"],
                    )

                    st.session_state["active_dataset_path"] = str(save_path)
                    st.session_state["active_dataset_name"] = uploaded.name
                    st.success(f"✅ Dataset **{uploaded.name}** saved and activated!")
                    st.balloons()

    # ── TAB 2: Schema ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Required Column Schema")
        st.markdown("Your CSV/Excel file **must** include all of the following columns (exact names).")

        schema = {
            "Column Name":        REQUIRED_COLUMNS,
            "Data Type":          ["String", "Float", "Float", "Float", "Float", "Float", "Integer", "Float", "Float", "Categorical"],
            "Valid Range / Notes":["Unique student ID", "0–100", "0–40", "0–60", "0–100",
                                   "0–168 (weekly hours)", "1–5 (integer scale)",
                                   "0.00–5.00", "-5.00 to 5.00", "Excellent / Good / Average / Poor / Fail"],
            "Required":           ["✅"] * 10,
        }
        st.dataframe(pd.DataFrame(schema), width='stretch', hide_index=True)

        st.markdown("---")
        st.markdown("##### 📥 Download Sample Template")
        sample_df = pd.DataFrame({
            "Student ID":          ["STU0001", "STU0002"],
            "Attendance":          [85.0, 60.0],
            "Assignment Score":    [78.0, 52.0],
            "Test Score":          [82.0, 49.0],
            "Study Hours":         [15.0, 6.0],
            "Class Participation": [4, 2],
            "Previous GPA":        [3.4, 2.1],
            "Performance Class":   ["Good", "Average"],
        })
        csv_bytes = sample_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️  Download CSV Template",
            data=csv_bytes,
            file_name="student_template.csv",
            mime="text/csv",
        )

    # ── TAB 3: Mock Data ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 🌱 Generate & Use Mock Dataset")
        st.markdown(
            "The system ships with a **200-row synthetic dataset** "
            "containing realistic distributions across all 5 performance classes. "
            "Use this to explore the system immediately without your own data."
        )

        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999, step=1)

        if st.button("🔄  Generate Mock Dataset", type="primary"):
            mock_df = generate_mock_dataset(seed=int(seed))
            st.success(f"✅ Generated **{len(mock_df)} rows** across 5 performance classes.")

            st.markdown("##### Preview (first 10 rows)")
            st.dataframe(mock_df.head(10), width='stretch', hide_index=True)

            st.markdown("##### Class Distribution")
            st.dataframe(
                mock_df["Performance Class"].value_counts().rename_axis("Class").reset_index(name="Count"),
                use_container_width=True, hide_index=True
            )

            # Save & activate
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            mock_path = UPLOADS_DIR / f"mock_data_seed{seed}.csv"
            mock_df.to_csv(str(mock_path), index=False)

            log_dataset(
                filename=mock_path.name,
                row_count=len(mock_df),
                col_count=len(mock_df.columns),
                file_path=str(mock_path),
                uploaded_by=user["username"],
            )
            st.session_state["active_dataset_path"] = str(mock_path)
            st.session_state["active_dataset_name"] = mock_path.name
            st.info(f"✅ Mock dataset activated as: **{mock_path.name}**")

            csv_dl = mock_df.to_csv(index=False).encode()
            st.download_button(
                label="⬇️  Download Mock Dataset",
                data=csv_dl,
                file_name=mock_path.name,
                mime="text/csv",
            )

    # ── Active Dataset Banner ─────────────────────────────────────────────────
    active = st.session_state.get("active_dataset_name")
    if active:
        st.markdown("---")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#14532d,#166534);
                    border:1px solid #22c55e44; border-radius:12px;
                    padding:0.85rem 1.2rem; margin-top:0.5rem;'>
            <span style='color:#4ade80; font-weight:600;'>✅ Active Dataset:</span>
            <span style='color:#e2e8f0; margin-left:0.5rem;'>{active}</span>
        </div>
        """, unsafe_allow_html=True)
