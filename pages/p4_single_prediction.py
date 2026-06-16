"""
p4_single_prediction.py — Single student prediction form.
Modes:
  1. Manual Entry     — single course / semester form (existing behaviour)
  2. Academic Timeline — upload multi-course CSV → cumulative CGPA forecast
Available to both Admin and Student roles.
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from core.auth      import require_auth, current_user
from core.ml_engine import (
    load_model, predict_single, model_is_trained,
    CLASS_COLOURS, estimate_cgpa,
)
from core.database  import log_prediction
from core.ml_engine import FEATURE_COLS

# ── Class metadata ─────────────────────────────────────────────────────────────
CLASS_META = {
    "Excellent": {"fg": "#22c55e", "bg": "#14532d", "icon": "🏆", "msg": "Outstanding academic achievement!"},
    "Good":      {"fg": "#3b82f6", "bg": "#1e3a5f", "icon": "👍", "msg": "Strong performance — keep it up!"},
    "Average":   {"fg": "#f59e0b", "bg": "#422006", "icon": "📘", "msg": "Meeting expectations. Room to grow!"},
    "Poor":      {"fg": "#f97316", "bg": "#431407", "icon": "⚠️",  "msg": "Below satisfactory — extra support recommended."},
    "Fail":      {"fg": "#ef4444", "bg": "#450a0a", "icon": "🚨", "msg": "Critical — immediate intervention required."},
}

# Grade-point mapping used by the J48 cumulative engine (5.0 scale)
GRADE_POINTS = {
    "Excellent": 5,
    "Good":      4,
    "Average":   3,
    "Poor":      2,
    "Fail":      0,
}

# ── Year-1 actual grade-point rule (no model — direct score mapping) ───────────
_SCORE_GP_THRESHOLDS = [
    (70, 5),   # >= 70  → Excellent (5)
    (60, 4),   # >= 60  → Good      (4)
    (50, 3),   # >= 50  → Average   (3)
    (45, 2),   # >= 45  → Poor      (2)
    (0,  0),   # <  45  → Fail      (0)
]

def _score_to_gp(total_score: float) -> tuple[int, str]:
    """Map a 0-100 Total_Score to (grade_point, class_label) using CGPA scale rules."""
    for threshold, gp in _SCORE_GP_THRESHOLDS:
        if total_score >= threshold:
            label = {5: "Excellent", 4: "Good", 3: "Average", 2: "Poor", 0: "Fail"}[gp]
            return gp, label
    return 0, "Fail"

# Timeline template — downloadable by the user
# Score format:
#   CA_Score   : Continuous Assessment mark out of 40  (e.g. 34/40)
#   Exam_Score : Final examination mark out of 60      (e.g. 52/60)
#   Total_Score: Computed automatically as CA + Exam   (max 100)
# Level column drives the phased validation mode: 100L, 200L, 300L, 400L
TIMELINE_TEMPLATE = pd.DataFrame({
    "Course_Code":  [
        "CSC101","MTH101","PHY101","GST101",   # 100L Sem 1
        "CSC102","MTH102","PHY102",             # 100L Sem 2
        "CSC201","MTH201","STA201","CSC202",   # 200L
        "CSC301","CSC302","MTH301","CSC303",   # 300L
        "CSC401","CSC402","CSC403","CSC404",   # 400L
    ],
    "Level":       [
        "100L","100L","100L","100L",
        "100L","100L","100L",
        "200L","200L","200L","200L",
        "300L","300L","300L","300L",
        "400L","400L","400L","400L",
    ],
    "Semester":    [1,1,1,1, 2,2,2, 1,1,1,2, 1,1,1,2, 1,1,2,2],
    "Credits":     [3,3,2,2, 3,3,2, 3,3,3,3, 3,3,3,3, 4,4,4,3],
    "Attendance_Pct": [
        85,72,90,88, 80,75,82,
        65,70,68,74,
        78,82,70,76,
        80,84,78,72,
    ],
    "CA_Score":    [34,28,38,35, 30,27,33, 22,26,24,29, 30,33,28,31, 35,36,30,28],  # /40
    "Exam_Score":  [52,44,56,50, 46,42,50, 36,44,40,46, 48,52,44,49, 54,55,47,43],  # /60
    "Study_Hours_Week": [12,10,14,11, 9,8,10, 8,10,9,11, 11,12,10,10, 13,14,11,10],
    "Class_Participation": [4,3,4,4, 3,3,3, 3,3,3,3, 3,4,3,3, 4,4,3,3],
})

# ── Timeline ingestion helper ──────────────────────────────────────────────────
_TIMELINE_REQUIRED = [
    "Course_Code", "Credits", "Attendance_Pct", "CA_Score", "Exam_Score",
]
_TIMELINE_OPTIONAL = {
    "Study_Hours_Week":    10.0,
    "Class_Participation": 3,
    "Semester":            1,
    "Level":               "?",
}


# Removed _map_timeline_to_features as we now use raw data and vectorised prediction.


# ── CGPA colour helper ─────────────────────────────────────────────────────────
def _cgpa_colour(cgpa: float) -> str:
    if cgpa >= 4.5:
        return "#22c55e"
    if cgpa >= 3.5:
        return "#3b82f6"
    if cgpa >= 2.5:
        return "#f59e0b"
    if cgpa >= 1.5:
        return "#f97316"
    return "#ef4444"


def _cgpa_class(cgpa: float) -> str:
    if cgpa >= 4.5: return "First Class"
    if cgpa >= 3.5: return "Second Class Upper"
    if cgpa >= 2.5: return "Second Class Lower"
    if cgpa >= 1.5: return "Third Class"
    return "Fail / Probation"


# ══════════════════════════════════════════════════════════════════════════════
def show():
    require_auth()
    user = current_user()
    role = user.get("role", "student")

    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0;'>
        <div class='section-header'>🔍 Single Student Prediction</div>
        <div class='section-sub'>Predict performance for a single course or forecast your full graduating CGPA</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_is_trained():
        st.warning("⚠️ No trained model found.")
        if role == "admin":
            st.info("Please go to the **Model Training** page and train the model first.")
        else:
            st.info("The prediction model has not been configured yet. Please contact your administrator.")
        return

    model = load_model()
    if model is None:
        st.error("❌ Failed to load the model. Please retrain on the Model Training page.")
        return

    # ── Mode selector ──────────────────────────────────────────────────────────
    mode = st.radio(
        "Prediction mode",
        options=["✏️  Manual Entry", "📂  Upload Academic Timeline"],
        horizontal=True,
        label_visibility="collapsed",
        key="pred_mode",
    )

    st.markdown("<hr style='border-color:#2d3555; margin:0.6rem 0 1.2rem 0;'/>",
                unsafe_allow_html=True)

    if mode == "✏️  Manual Entry":
        _render_manual(model, user)
    else:
        _render_timeline(model, user)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Manual Entry (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════
def _render_manual(model, user):
    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        st.markdown("#### 📝 Student Profile Entry")

        with st.form(key="prediction_form", clear_on_submit=False):
            student_id = st.text_input(
                "Student ID",
                placeholder="e.g. STU0042",
                help="Optional — used for logging and export.",
            )

            st.markdown("---")
            st.markdown("**Academic Metrics**")

            col1, col2 = st.columns(2)
            with col1:
                attendance = st.slider(
                    "Attendance (%)",
                    min_value=0.0, max_value=100.0, value=75.0, step=0.5,
                    help="Percentage of classes attended (0–100)"
                )
                ca_score = st.number_input(
                    "CA Score",
                    min_value=0.0, max_value=40.0, value=30.0, step=0.5,
                    help="Continuous Assessment score (0–40)"
                )
                exam_score = st.number_input(
                    "Exam Score",
                    min_value=0.0, max_value=60.0, value=45.0, step=0.5,
                    help="Examination score (0–60)"
                )

            with col2:
                study_hours = st.number_input(
                    "Study Hours / Week",
                    min_value=0.0, max_value=168.0, value=10.0, step=0.5,
                    help="Average weekly study hours"
                )
                class_participation = st.select_slider(
                    "Class Participation",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    format_func=lambda x: {
                        1: "1 — Very Low", 2: "2 — Low", 3: "3 — Moderate",
                        4: "4 — High",    5: "5 — Very High",
                    }[x],
                    help="Scale from 1 (lowest) to 5 (highest)"
                )
                previous_gpa = st.number_input(
                    "Previous GPA",
                    min_value=0.00, max_value=5.00, value=2.50, step=0.01,
                    format="%.2f",
                    help="GPA on 5.0 scale"
                )
                academic_momentum = st.number_input(
                    "Academic Momentum",
                    min_value=-5.00, max_value=5.00, value=0.00, step=0.01,
                    format="%.2f",
                    help="Current Previous GPA - Historical Previous GPA. Default to 0.0 if not applicable."
                )

            st.markdown("---")
            submitted = st.form_submit_button(
                "🔮  Predict Performance",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            total_score = ca_score + exam_score
            features = [
                attendance, ca_score, exam_score, total_score,
                study_hours, float(class_participation), previous_gpa,
                academic_momentum,
            ]
            with st.spinner("Analysing profile…"):
                prediction = predict_single(model, features)
                cgpa       = estimate_cgpa(prediction, previous_gpa)

            input_dict = {
                "Attendance_Pct":      attendance,
                "CA_Score":            ca_score,
                "Exam_Score":          exam_score,
                "Total_Score":         total_score,
                "Study_Hours_Week":    study_hours,
                "Class_Participation": class_participation,
                "Previous_GPA":        previous_gpa,
                "Academic_Momentum":   academic_momentum,
            }
            log_prediction(
                student_id   = student_id or "anonymous",
                input_data   = input_dict,
                predicted    = prediction,
                predicted_by = user["username"],
            )

            st.session_state["last_prediction"]       = prediction
            st.session_state["last_cgpa"]             = cgpa
            st.session_state["last_prediction_input"] = input_dict
            st.session_state["last_student_id"]       = student_id

    # ── Result panel ──────────────────────────────────────────────────────────
    with result_col:
        prediction = st.session_state.get("last_prediction")
        cgpa       = st.session_state.get("last_cgpa")
        inp        = st.session_state.get("last_prediction_input")
        sid        = st.session_state.get("last_student_id", "")

        if prediction is None:
            st.markdown("""
            <div style='background:#1e2538; border:1px solid #2d3555; border-radius:18px;
                        padding:2.5rem 2rem; text-align:center; margin-top:3rem;'>
                <div style='font-size:3rem; margin-bottom:0.8rem;'>🎓</div>
                <div style='color:#4a5580; font-size:1rem;'>
                    Fill in the form and click<br><b style='color:#7c8db5;'>Predict Performance</b>
                    <br>to see the result here.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            meta = CLASS_META.get(prediction, {"fg": "#888", "bg": "#222", "icon": "❓", "msg": ""})
            fg, bg, icon, msg = meta["fg"], meta["bg"], meta["icon"], meta["msg"]

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{bg},{bg}cc);
                        border:2px solid {fg}55; border-radius:20px;
                        padding:2rem 1.5rem; text-align:center; margin-top:1rem;'>
                <div style='font-size:3rem; margin-bottom:0.5rem;'>{icon}</div>
                <div style='font-size:0.8rem; color:{fg}99; letter-spacing:0.1em;
                            text-transform:uppercase; margin-bottom:0.3rem;'>
                    Predicted Performance
                </div>
                <div style='font-size:2.5rem; font-weight:800; color:{fg};
                            text-shadow: 0 0 30px {fg}66; margin-bottom:0.5rem;'>
                    {prediction}
                </div>
                <div style='color:{fg}bb; font-size:0.85rem; margin-bottom:1rem;'>
                    {msg}
                </div>
                <div style='background:#0d1117; border:1px solid {fg}33;
                            border-radius:12px; padding:0.8rem 1rem; margin-bottom:1rem;'>
                    <div style='font-size:0.7rem; color:{fg}88; letter-spacing:0.09em;
                                text-transform:uppercase; margin-bottom:0.25rem;'>
                        Estimated CGPA (5.0 scale)
                    </div>
                    <div style='font-size:2rem; font-weight:800; color:{fg};'>
                        {cgpa:.2f} <span style='font-size:0.9rem; font-weight:400;
                            color:{fg}77;'>/ 5.00</span>
                    </div>
                </div>
                {"<div style='color:#7c8db5; font-size:0.78rem;'>Student ID: " + sid + "</div>" if sid else ""}
            </div>
            """, unsafe_allow_html=True)

            if inp:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("##### 📋 Input Summary")
                st.dataframe(
                    pd.DataFrame({"Feature": list(inp.keys()), "Value": [str(v) for v in inp.values()]}),
                    use_container_width=True, hide_index=True,
                )

            try:
                proba       = model.predict_proba(np.array([list(inp.values())], dtype=float))[0]
                class_names = model.classes_

                st.markdown("##### 📊 Class Probabilities")
                st.dataframe(
                    pd.DataFrame({"Class": class_names, "Probability": [f"{p:.1%}" for p in proba]}),
                    use_container_width=True, hide_index=True,
                )

                fig, ax = plt.subplots(figsize=(5, 2.5))
                fig.patch.set_facecolor("#1e2130")
                ax.set_facecolor("#1e2130")
                colours = [CLASS_COLOURS.get(c, "#888") for c in class_names]
                ax.bar(class_names, proba * 100, color=colours, edgecolor="none", width=0.55)
                ax.set_ylabel("%", color="#a3b3d4", fontsize=9)
                ax.tick_params(colors="#a3b3d4", labelsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#2d3555")
                ax.spines["bottom"].set_color("#2d3555")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                st.markdown("---")
                with st.expander("🧠 Model Explainability (Why did the AI predict this?)"):
                    st.markdown("Feature importance weights driving this prediction:")
                    
                    if hasattr(model, "feature_importances_"):
                        fi_df = pd.DataFrame({
                            "Feature": FEATURE_COLS,
                            "Importance (%)": (model.feature_importances_ * 100).round(2),
                        }).sort_values("Importance (%)", ascending=True)
                        
                        fig_xai, ax_xai = plt.subplots(figsize=(6, 3))
                        fig_xai.patch.set_facecolor("#1e2130")
                        ax_xai.set_facecolor("#1e2130")
                        
                        bars = ax_xai.barh(fi_df["Feature"], fi_df["Importance (%)"], color="#3b82f6", edgecolor="none")
                        ax_xai.set_xlabel("Importance Weight (%)", color="#a3b3d4", fontsize=9)
                        ax_xai.tick_params(colors="#a3b3d4", labelsize=8)
                        ax_xai.spines["top"].set_visible(False)
                        ax_xai.spines["right"].set_visible(False)
                        ax_xai.spines["left"].set_color("#2d3555")
                        ax_xai.spines["bottom"].set_color("#2d3555")
                        fig_xai.tight_layout()
                        st.pyplot(fig_xai)
                        plt.close(fig_xai)
                    else:
                        st.info("Explainability is not available for this model type.")
                        
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Academic Timeline Upload & Phased CGPA Forecast
# ══════════════════════════════════════════════════════════════════════════════

# Canonical level ordering
_LEVEL_ORDER = ["100L", "200L", "300L", "400L"]


def _phased_forecast(df_raw: pd.DataFrame,
                     model) -> tuple[list[dict], pd.DataFrame]:
    """
    Run the phased year-by-year CGPA forecast using vectorised predictions.

    Phase 1 (100L)  — actual GPA via direct score→GP mapping; no J48.
    Phase N (200L+) — AI prediction with previous year's cumulative CGPA
                      threaded in as the 'Previous_GPA' feature for every
                      course in that level, alongside calculated Momentum.
    """
    present_levels = [lv for lv in _LEVEL_ORDER if lv in df_raw["Level"].values]
    extra = sorted(set(df_raw["Level"].values) - set(_LEVEL_ORDER))
    present_levels += extra

    df_out = df_raw.copy()
    
    # Ensure all required features exist with defaults
    for col in FEATURE_COLS:
        if col not in df_out.columns:
            df_out[col] = 0.0

    df_out["Phase"]                 = ""
    df_out["Predicted_Performance"] = ""
    df_out["Predicted_Grade_Point"] = 0
    df_out["Quality_Points"]        = 0.0
    df_out["Cumulative_CGPA_After"] = 0.0

    global_qp      = 0.0
    global_credits = 0.0
    prev_year_cgpa = 0.0

    phases = []

    for year_idx, level in enumerate(present_levels):
        mask   = df_out["Level"] == level
        idx    = df_out.index[mask]
        level_df = df_out.loc[idx].copy()

        is_yr1 = (year_idx == 0)

        # Broadcast calculated values
        if is_yr1:
            level_df['Previous_GPA'] = 0.0
            level_df['Academic_Momentum'] = 0.0
        else:
            level_df['Previous_GPA'] = round(prev_year_cgpa, 4)
            if year_idx == 1:
                momentum = 0.0
            else:
                historical_gpa = phases[year_idx - 2]["cum_cgpa"]
                momentum = round(prev_year_cgpa - historical_gpa, 4)
            level_df['Academic_Momentum'] = momentum

        # Write back to df_out for the record
        df_out.loc[idx, 'Previous_GPA'] = level_df['Previous_GPA']
        df_out.loc[idx, 'Academic_Momentum'] = level_df['Academic_Momentum']

        # Extract exactly the required features in the correct order
        X_predict = level_df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)

        if is_yr1:
            def apply_score_gp(row):
                return _score_to_gp(row['Total_Score'])
            
            res = level_df.apply(apply_score_gp, axis=1, result_type='expand')
            res.columns = ['gp', 'label']
            
            level_df["Predicted_Grade_Point"] = res["gp"].astype(int)
            level_df["Predicted_Performance"] = res["label"]
            phase_tag = "Actual"
        else:
            predictions = model.predict(X_predict.to_numpy(dtype=float))
            level_df["Predicted_Performance"] = predictions
            level_df["Predicted_Grade_Point"] = level_df["Predicted_Performance"].map(lambda x: GRADE_POINTS.get(x, 0)).astype(int)
            phase_tag = "Predicted"

        df_out.loc[idx, 'Predicted_Performance'] = level_df['Predicted_Performance']
        df_out.loc[idx, 'Predicted_Grade_Point'] = level_df['Predicted_Grade_Point']
        df_out.loc[idx, 'Phase'] = phase_tag

        cr = pd.to_numeric(level_df["Credits"], errors='coerce').fillna(0).astype(float)
        qp = level_df["Predicted_Grade_Point"] * cr

        df_out.loc[idx, 'Quality_Points'] = qp

        year_qp = qp.sum()
        year_credits = cr.sum()

        global_qp += year_qp
        global_credits += year_credits

        cum_cgpa = round(global_qp / global_credits, 4) if global_credits > 0 else 0.0
        year_cgpa = round(year_qp / year_credits, 2) if year_credits > 0 else 0.0

        df_out.loc[idx, 'Cumulative_CGPA_After'] = round(cum_cgpa, 2)

        course_rows = []
        for i, row in level_df.iterrows():
            course_rows.append({
                "Course_Code":   row.get("Course_Code", str(i)),
                "Semester":      row.get("Semester", ""),
                "Credits":       cr.loc[i],
                "Total_Score":   round(row.get("Total_Score", 0), 1),
                "Performance":   row["Predicted_Performance"],
                "Grade_Point":   row["Predicted_Grade_Point"],
                "Quality_Points": round(qp.loc[i], 1),
                "Cum_CGPA":      round(cum_cgpa, 2),
                "Phase":         phase_tag,
            })

        phases.append({
            "level":      level,
            "is_actual":  is_yr1,
            "year_cgpa":  year_cgpa,
            "cum_cgpa":   round(cum_cgpa, 2),
            "year_qp":    round(year_qp, 1),
            "year_cr":    int(year_credits),
            "course_df": pd.DataFrame(course_rows),
        })

        prev_year_cgpa = cum_cgpa

    return phases, df_out


def _render_timeline(model, user):
    # ── Instructions card ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#1a2040; border:1px solid #2d3555; border-radius:12px;
                padding:1rem 1.4rem; margin-bottom:1.2rem;'>
        <b style='color:#60a5fa;'>📋 Phased Year-by-Year CGPA Forecast:</b>
        <ol style='color:#a3b3d4; margin:0.5rem 0 0 0; font-size:0.88rem;'>
            <li><b>Year 1 (100L)</b> — actual GPA computed directly from scores; no model used.</li>
            <li><b>Year 2+ (200L–400L)</b> — AI model predicts each course; previous year's
                cumulative CGPA is threaded in as the <code>Previous GPA</code> feature. <code>Academic Momentum</code> is calculated automatically.</li>
            <li>Final Graduating CGPA = Σ all Quality Points ÷ Σ all Credits.</li>
        </ol>
        <div style='margin-top:0.9rem; background:#0d1117; border-radius:8px;
                    padding:0.7rem 1rem; font-size:0.82rem; color:#7c8db5;'>
            <b style='color:#60a5fa;'>Score format (Nigerian convention):</b><br/>
            &nbsp;• <code>CA_Score</code> — Continuous Assessment <b>out of 40</b><br/>
            &nbsp;• <code>Exam_Score</code> — Final Examination <b>out of 60</b> &nbsp;
              (system computes <code>Total_Score = CA + Exam</code> automatically)<br/>
            &nbsp;• <code>Level</code> must be one of: <code>100L, 200L, 300L, 400L</code><br/>
            &nbsp;• <code>Study_Hours_Week</code> and <code>Class_Participation</code> are optional
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_dl, _ = st.columns([2, 3])
    with col_dl:
        st.download_button(
            label="⬇️  Download 4-Year Timeline Template (CSV)",
            data=TIMELINE_TEMPLATE.to_csv(index=False).encode(),
            file_name="academic_timeline_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Academic Timeline",
        type=["csv", "xlsx"],
        key="timeline_uploader",
        help="One row per course. Must include a 'Level' column (100L, 200L, …).",
    )

    if not uploaded:
        st.markdown("""
        <div style='background:#1e2538; border:1px dashed #2d3555; border-radius:14px;
                    padding:2.5rem; text-align:center; margin-top:1rem;'>
            <div style='font-size:2.5rem; margin-bottom:0.6rem;'>📂</div>
            <div style='color:#4a5580; font-size:0.95rem;'>
                Upload your academic timeline CSV or Excel file above<br>
                to begin the phased CGPA forecast.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Read ──────────────────────────────────────────────────────────────────
    try:
        df_raw = (pd.read_excel(uploaded, engine="openpyxl")
                  if uploaded.name.endswith(".xlsx")
                  else pd.read_csv(uploaded))
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return

    # ── Validate ──────────────────────────────────────────────────────────────
    required = _TIMELINE_REQUIRED + ["Level"]
    missing  = [c for c in required if c not in df_raw.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.info("Download the template above to see the expected format.")
        return
    if len(df_raw) == 0:
        st.error("❌ The uploaded file contains no data rows.")
        return

    st.success(f"✅ File accepted — **{len(df_raw):,}** courses across "
               f"**{df_raw['Level'].nunique()}** academic level(s).")

    # ── Pre-process scores ────────────────────────────────────────────────────
    df_raw = df_raw.copy()
    df_raw["Total_Score"] = (
        pd.to_numeric(df_raw["CA_Score"],   errors="coerce").fillna(0)
        + pd.to_numeric(df_raw["Exam_Score"], errors="coerce").fillna(0)
    ).clip(0, 100)

    # ── Run phased engine ─────────────────────────────────────────────────────
    with st.spinner("Running phased year-by-year forecast…"):
        try:
            phases, df_out = _phased_forecast(df_raw, model)
        except Exception as e:
            st.error(f"❌ Forecast engine error: {e}")
            return

    final_cgpa   = phases[-1]["cum_cgpa"] if phases else 0.0
    final_colour = _cgpa_colour(final_cgpa)
    final_class  = _cgpa_class(final_cgpa)
    total_credits= sum(p["year_cr"] for p in phases)
    total_qp     = sum(p["year_qp"] for p in phases)

    # ═════════════════════════════════════════════════════════════════════════
    # 1. Headline graduating CGPA card
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0f1829,#1a2540);
                border:2px solid {final_colour}55; border-radius:24px;
                padding:2.5rem 2rem; text-align:center; margin-bottom:1.6rem;
                box-shadow:0 0 60px {final_colour}22;'>
        <div style='font-size:0.85rem; color:{final_colour}99; letter-spacing:0.14em;
                    text-transform:uppercase; margin-bottom:0.4rem;'>
            🎓 &nbsp; Forecasted Graduating CGPA
        </div>
        <div style='font-size:5.5rem; font-weight:900; color:{final_colour};
                    text-shadow:0 0 60px {final_colour}88; line-height:1;
                    margin-bottom:0.3rem;'>
            {final_cgpa:.2f}
        </div>
        <div style='font-size:1rem; color:{final_colour}cc; margin-bottom:0.8rem;'>out of 5.00</div>
        <div style='display:inline-block; background:{final_colour}22;
                    border:1px solid {final_colour}55; border-radius:20px;
                    padding:0.35rem 1.2rem; font-size:0.85rem;
                    color:{final_colour}; font-weight:600; letter-spacing:0.05em;'>
            {final_class}
        </div>
        <div style='margin-top:1rem; font-size:0.78rem; color:#4a5580;'>
            {len(df_out)} courses &nbsp;·&nbsp; {int(total_credits)} credit units
            &nbsp;·&nbsp; {total_qp:.0f} quality points
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # 2. Year-strip — CGPA per level at a glance
    # ═════════════════════════════════════════════════════════════════════════
    strip_cols = st.columns(len(phases))
    for col_ui, ph in zip(strip_cols, phases):
        clr   = _cgpa_colour(ph["cum_cgpa"])
        badge = "✅ Actual" if ph["is_actual"] else "🔮 Predicted"
        with col_ui:
            st.markdown(f"""
            <div style='background:#1e2538; border:1px solid {clr}44;
                        border-radius:14px; padding:1rem 0.8rem; text-align:center;'>
                <div style='font-size:0.65rem; color:{clr}88; letter-spacing:0.08em;
                            text-transform:uppercase; margin-bottom:0.2rem;'>{ph['level']}</div>
                <div style='font-size:1.9rem; font-weight:800; color:{clr};
                            line-height:1; margin-bottom:0.2rem;'>{ph['cum_cgpa']:.2f}</div>
                <div style='font-size:0.68rem; color:{clr}99; margin-bottom:0.3rem;'>Cum. CGPA</div>
                <div style='display:inline-block; background:{clr}18;
                            border:1px solid {clr}44; border-radius:10px;
                            padding:0.15rem 0.6rem; font-size:0.65rem; color:{clr};'>{badge}</div>
            </div>
            """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. Per-level expander cards
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### 📚 Year-by-Year Breakdown")

    for ph in phases:
        level    = ph["level"]
        clr      = _cgpa_colour(ph["cum_cgpa"])
        is_act   = ph["is_actual"]
        label_yr = "Actual GPA (base truth)" if is_act else "Predicted Cumulative CGPA"
        icon_yr  = "📐" if is_act else "🔮"

        with st.expander(
            f"{icon_yr}  {level}  —  {label_yr}: **{ph['cum_cgpa']:.2f}**",
            expanded=(ph == phases[0]),   # expand Year 1 by default
        ):
            # Mini metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div style='background:#1e2538; border:1px solid {clr}33;
                            border-radius:10px; padding:0.6rem 0.8rem; text-align:center;'>
                    <div style='font-size:0.62rem; color:#7c8db5; text-transform:uppercase;'>Year CGPA</div>
                    <div style='font-size:1.3rem; font-weight:700; color:{clr};'>{ph['year_cgpa']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div style='background:#1e2538; border:1px solid {clr}33;
                            border-radius:10px; padding:0.6rem 0.8rem; text-align:center;'>
                    <div style='font-size:0.62rem; color:#7c8db5; text-transform:uppercase;'>Cum. CGPA</div>
                    <div style='font-size:1.3rem; font-weight:700; color:{clr};'>{ph['cum_cgpa']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div style='background:#1e2538; border:1px solid #3b82f633;
                            border-radius:10px; padding:0.6rem 0.8rem; text-align:center;'>
                    <div style='font-size:0.62rem; color:#7c8db5; text-transform:uppercase;'>Credits</div>
                    <div style='font-size:1.3rem; font-weight:700; color:#3b82f6;'>{ph['year_cr']}</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div style='background:#1e2538; border:1px solid #6366f133;
                            border-radius:10px; padding:0.6rem 0.8rem; text-align:center;'>
                    <div style='font-size:0.62rem; color:#7c8db5; text-transform:uppercase;'>Quality Pts</div>
                    <div style='font-size:1.3rem; font-weight:700; color:#6366f1;'>{ph['year_qp']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            # Course table — style Performance column by class colour
            cdf = ph["course_df"].copy()
            if "Phase" in cdf.columns:
                cdf = cdf.drop(columns=["Phase"])

            def _style_perf(col):
                if col.name == "Performance":
                    return [f"color:{CLASS_COLOURS.get(v,'#e2e8f0')};font-weight:600" for v in col]
                return ["" for _ in col]

            st.dataframe(
                cdf.style.apply(_style_perf, axis=0),
                use_container_width=True,
                hide_index=True,
                height=min(350, 38 + len(cdf) * 36),
            )

            if is_act:
                st.caption(
                    "ℹ️ Year 1 grades are calculated from actual scores using the standard "
                    "5.0 scale (≥70→5, 60–69→4, 50–59→3, 45–49→2, <45→0). "
                    "No machine learning model is used for this baseline year."
                )
            else:
                prev = phases[phases.index(ph) - 1]
                st.caption(
                    f"🔮 AI model predictions used. Previous year's cumulative CGPA "
                    f"(**{prev['cum_cgpa']:.2f}**) was passed as the Previous GPA feature "
                    f"for every {level} course."
                )

    # ═════════════════════════════════════════════════════════════════════════
    # 4. Global trend chart (Cumulative CGPA after each Level)
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("##### 📈 Cumulative CGPA Trajectory (Year-by-Year)")

    level_names = [p["level"] for p in phases]
    level_cgpas = [p["cum_cgpa"] for p in phases]
    level_phases = ["Actual" if p["is_actual"] else "Predicted" for p in phases]

    fig, ax = plt.subplots(figsize=(max(7, len(level_names) * 1.5), 4))
    fig.patch.set_facecolor("#1a1f2e")
    ax.set_facecolor("#1a1f2e")

    xs = list(range(len(level_names)))

    # Shade actual (Phase 1) vs predicted regions
    act_end = sum(1 for p in level_phases if p == "Actual") - 1
    if act_end >= 0:
        ax.axvspan(-0.5, act_end + 0.5, alpha=0.06, color="#22c55e", label="_nolegend_")
        ax.text(act_end / 2, 5.0, "Actual", ha="center", va="top",
                color="#22c55e", fontsize=8, alpha=0.7)
    if act_end < len(level_names) - 1:
        ax.axvspan(act_end + 0.5, len(level_names) - 0.5, alpha=0.05,
                   color="#3b82f6", label="_nolegend_")
        ax.text((act_end + len(level_names)) / 2, 5.0, "Predicted", ha="center", va="top",
                color="#3b82f6", fontsize=8, alpha=0.7)

    # Reference grade boundaries
    for ref, lbl in [(4.5, "1st Class"), (3.5, "2nd Upper"),
                     (2.5, "2nd Lower"), (1.5, "3rd Class")]:
        ax.axhline(ref, color="#2d3555", linewidth=0.8, linestyle="--")
        ax.text(len(xs) - 0.5, ref + 0.07, lbl,
                ha="right", va="bottom", color="#3a4565", fontsize=8)

    ax.fill_between(xs, level_cgpas, alpha=0.12, color=final_colour)
    ax.plot(xs, level_cgpas, color=final_colour, linewidth=2.5, zorder=3)

    # Dots coloured by phase
    for xi, (yv, phase) in enumerate(zip(level_cgpas, level_phases)):
        dot_clr = "#22c55e" if phase == "Actual" else "#3b82f6"
        marker  = "D" if phase == "Actual" else "o"
        ax.scatter(xi, yv, color=dot_clr, s=80, zorder=4,
                   edgecolors="#1a1f2e", linewidths=1.5, marker=marker)

    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(0, 5.3)
    ax.set_xticks(xs)
    ax.set_xticklabels(level_names, color="#7c8db5", fontsize=9)
    ax.set_ylabel("Cumulative CGPA", color="#a3b3d4", fontsize=9)
    ax.set_title("End-of-Year Cumulative CGPA Trajectory",
                 color="#e2e8f0", fontsize=10, pad=10)
    ax.tick_params(colors="#a3b3d4", labelsize=8)
    ax.spines["top"].set_visible(False);   ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3555"); ax.spines["bottom"].set_color("#2d3555")

    legend_items = [
        plt.Line2D([0],[0], marker="D", color="#22c55e", markersize=6,
                   label="Actual Base Truth", linestyle="None"),
        plt.Line2D([0],[0], marker="o", color="#3b82f6", markersize=6,
                   label="Predicted Forecast", linestyle="None"),
    ]
    ax.legend(handles=legend_items, loc="lower right", framealpha=0.15,
              labelcolor="#c4cfea", fontsize=8, edgecolor="#2d3555")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ═════════════════════════════════════════════════════════════════════════
    # 5. Full breakdown table + downloads
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br/>", unsafe_allow_html=True)
    with st.expander("📋 Full Course-by-Course Data Table", expanded=False):
        display_cols = [c for c in [
            "Course_Code", "Level", "Semester", "Credits",
            "CA_Score", "Exam_Score", "Total_Score",
            "Phase", "Predicted_Performance", "Predicted_Grade_Point",
            "Quality_Points", "Cumulative_CGPA_After",
        ] if c in df_out.columns]

        def _colour_phase(col):
            if col.name == "Predicted_Performance":
                return [f"color:{CLASS_COLOURS.get(v,'#e2e8f0')};font-weight:600" for v in col]
            if col.name == "Phase":
                return [
                    "color:#22c55e;font-weight:500" if v == "Actual" else "color:#3b82f6;"
                    for v in col
                ]
            return ["" for _ in col]

        st.dataframe(
            df_out[display_cols].style.apply(_colour_phase, axis=0),
            use_container_width=True,
            hide_index=True,
            height=min(500, 38 + len(df_out) * 35),
        )

        csv_bytes = df_out[display_cols].to_csv(index=False).encode()
        st.download_button(
            "⬇️  Download as CSV",
            csv_bytes,
            file_name="phased_cgpa_forecast.csv",
            mime="text/csv",
        )
        buf = io.BytesIO()
        df_out[display_cols].to_excel(buf, index=False, engine="openpyxl")
        st.download_button(
            "⬇️  Download as Excel",
            buf.getvalue(),
            file_name="phased_cgpa_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 6. Early Warning Dashboard
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### 🚨 Academic Intervention Dashboard")
    
    at_risk_df = df_out[df_out["Predicted_Performance"].isin(["Poor", "Fail"])]
    if len(at_risk_df) > 0:
        with st.container(border=True):
            st.error(f"**Action Required:** {len(at_risk_df)} course(s) flagged as high-risk.")
            display_cols_risk = ["Level", "Semester", "Course_Code", "Predicted_Performance"]
            st.dataframe(at_risk_df[display_cols_risk], use_container_width=True, hide_index=True)
    else:
        st.success("✅ Student is not currently at risk of failing any predicted courses. Excellent trajectory!")

    # ═════════════════════════════════════════════════════════════════════════
    # 7. Automated PDF Report
    # ═════════════════════════════════════════════════════════════════════════
    from core.reports import generate_academic_report
    st.markdown("<br/>", unsafe_allow_html=True)
    try:
        pdf_bytes = generate_academic_report(df_out, final_cgpa)
        st.download_button(
            label="📄 Download Official Academic Forecast Report (PDF)",
            data=pdf_bytes,
            file_name="Academic_Forecast_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Could not generate PDF report: {e}")
