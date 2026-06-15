"""
fix_deprecations.py
Replace st.dataframe(..., use_container_width=True) with st.dataframe(..., width='stretch')
Only for st.dataframe / st.data_editor - NOT st.button (use_container_width still valid there).
"""
import re, os

files = [
    "app.py",
    "pages/p1_dashboard.py",
    "pages/p2_dataset_upload.py",
    "pages/p3_model_training.py",
    "pages/p4_single_prediction.py",
    "pages/p5_batch_prediction.py",
]

# Pattern: replace use_container_width=True only inside st.dataframe calls
# Simple approach: replace the kwarg globally in dataframe context
def fix_file(path):
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    # Replace use_container_width in st.dataframe() and st.data_editor() calls
    # We do a line-by-line scan and only replace on lines that contain st.dataframe
    lines = original.split("\n")
    new_lines = []
    changed = 0
    for line in lines:
        if "st.dataframe" in line or "st.data_editor" in line:
            new_line = line.replace(", use_container_width=True", ", width='stretch'")
            new_line = new_line.replace(", use_container_width=False", ", width='content'")
            new_line = new_line.replace(",use_container_width=True", ", width='stretch'")
            new_line = new_line.replace(",use_container_width=False", ", width='content'")
            if new_line != line:
                changed += 1
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    new_content = "\n".join(new_lines)
    if new_content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  Fixed ({changed} lines): {path}")
    else:
        print(f"  No changes:              {path}")

for f in files:
    if os.path.exists(f):
        fix_file(f)
    else:
        print(f"  NOT FOUND: {f}")

print("\nDone.")
