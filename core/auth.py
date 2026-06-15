"""
auth.py — Session-based authentication and role-gating helpers.
Uses SQLite via database.py for credential verification.
"""

import streamlit as st
from core.database import verify_user


def login_page():
    """Render a full-screen login form. Returns True if authenticated."""
    # ── Page chrome ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        /* hide sidebar on login */
        [data-testid="stSidebar"] {display: none;}

        .login-card {
            max-width: 440px;
            margin: 6rem auto 0 auto;
            background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
            border: 1px solid #3a4060;
            border-radius: 20px;
            padding: 3rem 2.5rem;
            box-shadow: 0 24px 64px rgba(0,0,0,0.5);
        }
        .login-logo {
            font-size: 3rem;
            text-align: center;
            margin-bottom: 0.25rem;
        }
        .login-title {
            font-size: 1.6rem;
            font-weight: 700;
            text-align: center;
            color: #e2e8f0;
            margin-bottom: 0.3rem;
        }
        .login-subtitle {
            text-align: center;
            color: #7c8db5;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        .role-hint {
            background: #1a1f2e;
            border: 1px solid #2d3555;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 0.78rem;
            color: #7c8db5;
            margin-top: 1.5rem;
        }
        .role-hint strong { color: #a3b3d4; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-logo">🎓</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-title">AcademiQ Predict</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Student Performance Prediction System</div>', unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username", key="login_user")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_btn = st.button("🔐  Sign In", use_container_width=True, type="primary")

    if login_btn:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            user = verify_user(username, password)
            if user:
                st.session_state["user"] = {
                    "username": user["username"],
                    "role":     user["role"],
                    "id":       user["id"],
                }
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

    st.markdown("""
    <div class="role-hint">
        <strong>Default Accounts</strong><br>
        Admin: &nbsp;<code>admin</code> / <code>admin123</code><br>
        Student: <code>student</code> / <code>student123</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return False


def require_auth():
    """Call at the top of every page. Redirects to login if not authenticated."""
    if "user" not in st.session_state or not st.session_state["user"]:
        login_page()
        st.stop()


def require_admin():
    """Call on admin-only pages. Shows an error and stops if role != admin."""
    require_auth()
    if st.session_state["user"]["role"] != "admin":
        st.error("🚫 Administrator access required for this section.")
        st.info("Please log in with an administrator account.")
        st.stop()


def current_user() -> dict:
    return st.session_state.get("user", {})


def logout():
    st.session_state.pop("user", None)
    st.rerun()
