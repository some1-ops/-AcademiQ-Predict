import ast
import sys

files = [
    "core/database.py",
    "core/auth.py",
    "core/mock_data.py",
    "core/data_validator.py",
    "core/ml_engine.py",
    "app.py",
    "pages/p1_dashboard.py",
    "pages/p2_dataset_upload.py",
    "pages/p3_model_training.py",
    "pages/p4_single_prediction.py",
    "pages/p5_batch_prediction.py",
]

all_ok = True
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fh:
            source = fh.read()
        ast.parse(source)
        print(f"  OK  {f}")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR  {f}  line {e.lineno}: {e.msg}")
        all_ok = False
    except Exception as e:
        print(f"  ERROR  {f}: {e}")
        all_ok = False

if all_ok:
    print("\nAll files passed syntax check.")
else:
    print("\nSome files have errors.")
    sys.exit(1)
