"""
p4_single_prediction.py — Single student manual-entry prediction form.
Available to both Admin and Student roles.
"""

import streamlit as st
from core.auth      import require_auth, current_user
from core.ml_engine import load_model, predict_single, model_is_trained, CLASS_COLOURS
from core.database  import log_prediction


# ── Colour metadata ────────────────────────────────────────────────────────────
CLASS_META = {
    "Excellent": {"fg": "#22c55e", "bg": "#14532d", "icon": "🏆", "msg": "Outstanding academic achievement!"},
    "Good":      {"fg": "#3b82f6", "bg": "#1e3a5f", "icon": "👍", "msg": "Strong performance — keep it up!"},
    "Average":   {"fg": "#f59e0b", "bg": "#422006", "icon": "📘", "msg": "Meeting expectations. Room to grow!"},
    "Poor":      {"fg": "#f97316", "bg": "#431407", "icon": "⚠️",  "msg": "Below satisfactory — extra support recommended."},
    "Fail":      {"fg": "#ef4444", "bg": "#450a0a", "icon": "🚨", "msg": "Critical — immediate intervention required."},
}


def show():
    require_auth()
    user = current_user()
    role = user.get("role", "student")

    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0;'>
        <div class='section-header'>🔍 Single Student Prediction</div>
        <div class='section-sub'>Enter a student's academic metrics and receive an instant performance classification</div>
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

    # ── Layout ────────────────────────────────────────────────────────────────
    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        st.markdown("#### 📝 Student Profile Entry")

        with st.form(key="prediction_form", clear_on_submit=False):
            # Student ID
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
                assignment_score = st.number_input(
                    "Assignment Score",
                    min_value=0.0, max_value=100.0, value=70.0, step=0.5,
                    help="Total assignment score (0–100)"
                )
                test_score = st.number_input(
                    "Test Score",
                    min_value=0.0, max_value=100.0, value=65.0, step=0.5,
                    help="Examination/test score (0–100)"
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
                        1: "1 — Very Low",
                        2: "2 — Low",
                        3: "3 — Moderate",
                        4: "4 — High",
                        5: "5 — Very High",
                    }[x],
                    help="Scale from 1 (lowest) to 5 (highest)"
                )
                previous_gpa = st.number_input(
                    "Previous GPA",
                    min_value=0.00, max_value=4.00, value=2.50, step=0.01,
                    format="%.2f",
                    help="GPA on 4.0 scale"
                )

            st.markdown("---")
            submitted = st.form_submit_button(
                "🔮  Predict Performance",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            features = [
                attendance,
                assignment_score,
                test_score,
                study_hours,
                float(class_participation),
                previous_gpa,
            ]

            with st.spinner("Analysing profile…"):
                prediction = predict_single(model, features)

            # Log prediction
            input_dict = {
                "Attendance":          attendance,
                "Assignment Score":    assignment_score,
                "Test Score":          test_score,
                "Study Hours":         study_hours,
                "Class Participation": class_participation,
                "Previous GPA":        previous_gpa,
            }
            log_prediction(
                student_id   = student_id or "anonymous",
                input_data   = input_dict,
                predicted    = prediction,
                predicted_by = user["username"],
            )

            st.session_state["last_prediction"]       = prediction
            st.session_state["last_prediction_input"] = input_dict
            st.session_state["last_student_id"]       = student_id

    # ── Result Panel ──────────────────────────────────────────────────────────
    with result_col:
        prediction = st.session_state.get("last_prediction")
        inp        = st.session_state.get("last_prediction_input")
        sid        = st.session_state.get("last_student_id", "")

        if prediction is None:
            # Placeholder
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
            meta = CLASS_META.get(prediction, {"fg": "#888", "bg": "#222",
                                               "icon": "❓", "msg": ""})
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
                <div style='color:{fg}bb; font-size:0.85rem; margin-bottom:1.2rem;'>
                    {msg}
                </div>
                {"<div style='color:#7c8db5; font-size:0.78rem;'>Student ID: " + sid + "</div>" if sid else ""}
            </div>
            """, unsafe_allow_html=True)

            if inp:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("##### 📋 Input Summary")
                summary_df = {
                    "Feature": list(inp.keys()),
                    "Value":   [str(v) for v in inp.values()],
                }
                import pandas as pd
                st.dataframe(pd.DataFrame(summary_df),
                             use_container_width=True, hide_index=True)

            # ── Try all-class probabilities (if available) ────────────────
            try:
                import numpy as np
                features_arr = list(inp.values())
                import numpy as _np
                proba = model.predict_proba(_np.array([features_arr], dtype=float))[0]
                class_names = model.classes_

                st.markdown("##### 📊 Class Probabilities")
                prob_df = {
                    "Class":       class_names,
                    "Probability": [f"{p:.1%}" for p in proba],
                }
                import pandas as pd
                st.dataframe(pd.DataFrame(prob_df),
                             use_container_width=True, hide_index=True)

                # Mini bar chart
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

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
            except Exception:
                pass
