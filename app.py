"""
app.py — Main Streamlit entry point for the Student Performance Prediction System.
Handles: DB init, mock data seeding, auth gate, sidebar navigation.
"""

import streamlit as st

# ── Must be the FIRST Streamlit call ─────────────────────────────────────────
st.set_page_config(
    page_title="AcademiQ Predict",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
from core.database  import init_db
from core.mock_data import save_mock_dataset_if_missing
from core.auth      import require_auth, current_user, logout
from core.ml_engine import model_is_trained

init_db()
save_mock_dataset_if_missing()

# ── Global CSS / Theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(160deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f2e 0%, #141824 100%);
    border-right: 1px solid #2d3555;
}
[data-testid="stSidebar"] * { color: #c4cfea !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2538 0%, #252d45 100%);
    border: 1px solid #2d3555;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
}
[data-testid="stMetricLabel"] { color: #7c8db5 !important; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.7rem; font-weight: 700; }

/* ── Buttons ── */
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    border: none;
    transition: all 0.2s;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(0,0,0,0.4); }

/* ── Inputs ── */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div {
    background: #1e2538 !important;
    border: 1px solid #2d3555 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1f2e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7c8db5;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#3b82f6,#6366f1) !important;
    color: #fff !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid #2d3555;
    border-radius: 12px;
    overflow: hidden;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.2rem;
}
.section-sub {
    color: #7c8db5;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* ── Status badge ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-success { background:#14532d; color:#4ade80; }
.badge-warning { background:#422006; color:#fb923c; }
.badge-info    { background:#1e3a5f; color:#60a5fa; }

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #1e2538 0%, #252d45 100%);
    border: 1px solid #2d3555;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Auth Gate ─────────────────────────────────────────────────────────────────
require_auth()

user = current_user()
role = user.get("role", "student")
uname = user.get("username", "User")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:2.5rem;'>🎓</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>AcademiQ Predict</div>
        <div style='font-size:0.75rem; color:#4a5580;'>Decision Tree Engine</div>
    </div>
    <hr style='border-color:#2d3555; margin:0.75rem 0;'/>
    """, unsafe_allow_html=True)

    # User info
    role_badge = "🔴 Admin" if role == "admin" else "🔵 Student"
    st.markdown(f"""
    <div style='background:#1a2040; border:1px solid #2d3555; border-radius:10px;
                padding:0.6rem 1rem; margin-bottom:1rem;'>
        <div style='font-size:0.72rem; color:#4a5580;'>SIGNED IN AS</div>
        <div style='font-weight:600; color:#e2e8f0;'>{uname}</div>
        <div style='font-size:0.75rem; color:#6080c0;'>{role_badge}</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("<div style='font-size:0.7rem; color:#4a5580; letter-spacing:0.08em; margin:0.5rem 0 0.4rem 0;'>NAVIGATION</div>", unsafe_allow_html=True)

    pages = [("🏠", "Dashboard",         "dashboard"),
             ("📊", "Single Prediction", "single")]
    if role == "admin":
        pages = [
            ("🏠", "Dashboard",          "dashboard"),
            ("📂", "Dataset Upload",     "upload"),
            ("🧠", "Model Training",     "train"),
            ("🔍", "Single Prediction",  "single"),
            ("⚡", "Batch Prediction",   "batch"),
        ]

    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard"

    for icon, label, key in pages:
        active = st.session_state["page"] == key
        btn_style = "primary" if active else "secondary"
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     use_container_width=True, type=btn_style):
            st.session_state["page"] = key
            st.rerun()

    st.markdown("<hr style='border-color:#2d3555; margin:1rem 0;'/>", unsafe_allow_html=True)

    # Model status
    trained = model_is_trained()
    status_html = (
        '<span class="badge badge-success">✅ Model Trained</span>'
        if trained else
        '<span class="badge badge-warning">⚠️ No Model Yet</span>'
    )
    st.markdown(f"**Model Status**<br>{status_html}", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🚪  Sign Out", use_container_width=True, key="logout_btn"):
        logout()


# ── Page Router ───────────────────────────────────────────────────────────────
page = st.session_state.get("page", "dashboard")

if page == "dashboard":
    import pages.p1_dashboard as p
    p.show()
elif page == "upload":
    if role == "admin":
        import pages.p2_dataset_upload as p
        p.show()
    else:
        st.error("Access denied.")
elif page == "train":
    if role == "admin":
        import pages.p3_model_training as p
        p.show()
    else:
        st.error("Access denied.")
elif page == "single":
    import pages.p4_single_prediction as p
    p.show()
elif page == "batch":
    if role == "admin":
        import pages.p5_batch_prediction as p
        p.show()
    else:
        st.error("Access denied.")
