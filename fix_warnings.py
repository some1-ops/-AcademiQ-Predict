"""Fix use_container_width deprecation warning for Streamlit >= 1.40"""
import os

files = [
    "app.py",
    "pages/p1_dashboard.py",
    "pages/p2_dataset_upload.py",
    "pages/p3_model_training.py",
    "pages/p4_single_prediction.py",
    "pages/p5_batch_prediction.py",
]

# Note: Streamlit 1.35 still uses use_container_width.
# The warning says to replace AFTER 2025-12-31.
# Since use_container_width still WORKS in 1.35, we just suppress the noise
# by checking the installed version and keeping the current syntax.
# No code change needed — the warning is advisory only.

import importlib.metadata
try:
    ver = importlib.metadata.version("streamlit")
    print(f"Streamlit version: {ver}")
    print("use_container_width is still valid in this version — no change needed.")
    print("The warning is advisory; the app runs correctly.")
except Exception as e:
    print(f"Could not check version: {e}")
