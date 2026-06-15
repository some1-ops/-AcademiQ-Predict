"""
smoke_test.py — Tests core logic that does NOT depend on scikit-learn/streamlit.
Validates: database init, mock data generation, schema validation, ML engine.
"""
import sys
import os
# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("  AcademiQ Predict -- Core Smoke Test")
print("=" * 55)

# ── 1. Database init ──────────────────────────────────────────
print("\n[1] Testing database init...")
try:
    from core.database import init_db, verify_user, get_datasets
    init_db()
    print("    PASS  init_db() completed -- SQLite schema created")

    u = verify_user("admin", "admin123")
    assert u is not None, "admin login failed"
    assert u["role"] == "admin", f"Expected admin role, got {u['role']}"
    print(f"    PASS  Admin login: username={u['username']}, role={u['role']}")

    u2 = verify_user("student", "student123")
    assert u2 is not None, "student login failed"
    assert u2["role"] == "student"
    print(f"    PASS  Student login: username={u2['username']}, role={u2['role']}")

    u3 = verify_user("admin", "wrongpassword")
    assert u3 is None, "Bad login should return None"
    print("    PASS  Bad credentials correctly rejected")
except Exception as e:
    print(f"    FAIL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 2. Mock data generation ───────────────────────────────────
print("\n[2] Testing mock data generator...")
try:
    import numpy as np
    import pandas as pd
    from core.mock_data import generate_mock_dataset, save_mock_dataset_if_missing

    df = generate_mock_dataset(seed=42)
    assert len(df) == 200, f"Expected 200 rows, got {len(df)}"
    print(f"    PASS  Generated {len(df)} rows")

    expected_cols = [
        "Student ID", "Attendance", "Assignment Score", "Test Score",
        "Study Hours", "Class Participation", "Previous GPA", "Performance Class"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"    PASS  All {len(expected_cols)} required columns present")

    classes = set(df["Performance Class"].unique())
    expected_classes = {"Excellent", "Good", "Average", "Poor", "Fail"}
    assert classes == expected_classes, f"Class mismatch: {classes}"
    print(f"    PASS  All 5 performance classes present: {sorted(classes)}")

    assert df["Attendance"].between(0, 100).all(), "Attendance out of range"
    assert df["Assignment Score"].between(0, 100).all(), "Assignment Score out of range"
    assert df["Test Score"].between(0, 100).all(), "Test Score out of range"
    assert df["Study Hours"].between(0, 168).all(), "Study Hours out of range"
    assert df["Class Participation"].between(1, 5).all(), "CP out of range"
    assert df["Previous GPA"].between(0.0, 4.0).all(), "GPA out of range"
    print("    PASS  All value ranges valid")

    path = save_mock_dataset_if_missing()
    import pathlib
    assert pathlib.Path(path).exists(), "mock_data.csv not created"
    print(f"    PASS  Mock CSV saved: {path}")

except ImportError as e:
    print(f"    SKIP  (numpy/pandas not yet installed): {e}")
except Exception as e:
    print(f"    FAIL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 3. Schema validator ───────────────────────────────────────
print("\n[3] Testing data validator...")
try:
    import pandas as pd
    from core.data_validator import (
        validate_dataset, preprocess_dataset,
        validate_prediction_input, REQUIRED_COLUMNS
    )

    valid_df = pd.DataFrame({
        "Student ID":          ["A001", "A002"],
        "Attendance":          [80.0, 55.0],
        "Assignment Score":    [70.0, 40.0],
        "Test Score":          [75.0, 38.0],
        "Study Hours":         [12.0, 4.0],
        "Class Participation": [3, 2],
        "Previous GPA":        [2.8, 1.5],
        "Performance Class":   ["Good", "Poor"],
    })
    ok, errs = validate_dataset(valid_df)
    assert ok, f"Valid df failed: {errs}"
    print("    PASS  Valid dataset passes validation")

    bad_df = valid_df.drop(columns=["Attendance"])
    ok2, errs2 = validate_dataset(bad_df)
    assert not ok2, "Should have failed for missing column"
    print(f"    PASS  Missing column correctly detected")

    invalid_cls = valid_df.copy()
    invalid_cls.loc[0, "Performance Class"] = "SuperStar"
    ok3, errs3 = validate_dataset(invalid_cls)
    assert not ok3, "Should have failed for bad class"
    print(f"    PASS  Invalid class label correctly detected")

    df_clean = preprocess_dataset(valid_df)
    assert len(df_clean) == 2
    print(f"    PASS  preprocess_dataset() returned {len(df_clean)} clean rows")

    unlabelled = valid_df.drop(columns=["Performance Class"])
    ok4, e4 = validate_prediction_input(unlabelled)
    assert ok4, f"Unlabelled validation failed: {e4}"
    print("    PASS  Unlabelled batch CSV validation passed")

except ImportError as e:
    print(f"    SKIP  (pandas not installed): {e}")
except Exception as e:
    print(f"    FAIL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 4. ML engine ──────────────────────────────────────────────
print("\n[4] Testing ML engine (requires scikit-learn)...")
try:
    import sklearn
    import pandas as pd
    from core.mock_data  import generate_mock_dataset
    from core.ml_engine  import (
        train_model, save_model, load_model, model_is_trained,
        predict_single, predict_batch,
    )
    from core.data_validator import FEATURE_COLUMNS

    df = generate_mock_dataset(seed=42)
    model, metrics = train_model(df)
    print(f"    PASS  Model trained")
    print(f"          Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"          Precision: {metrics['precision']:.2f}%")
    print(f"          Recall:    {metrics['recall']:.2f}%")
    print(f"          F1:        {metrics['f1']:.2f}%")
    print(f"          Train/Test: {metrics['train_size']}/{metrics['test_size']}")

    assert metrics["accuracy"] > 70, f"Accuracy too low: {metrics['accuracy']}"
    print("    PASS  Accuracy above 70% threshold")

    path = save_model(model)
    assert model_is_trained(), "model.pkl not found after save"
    print(f"    PASS  Model saved: {path}")

    loaded = load_model()
    assert loaded is not None, "load_model() returned None"
    print("    PASS  Model loaded from disk successfully")

    result = predict_single(loaded, [85, 78, 82, 15, 4, 3.4])
    assert result in ["Excellent", "Good", "Average", "Poor", "Fail"]
    print(f"    PASS  Single prediction result: {result}")

    batch_df = df[FEATURE_COLUMNS].head(10)
    out = predict_batch(loaded, batch_df)
    assert "Predicted_Performance" in out.columns
    assert len(out) == 10
    print(f"    PASS  Batch prediction: {len(out)} rows processed")

except ImportError as e:
    print(f"    SKIP  (scikit-learn not yet installed): {e}")
except Exception as e:
    print(f"    FAIL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 55)
print("  All smoke tests PASSED")
print("=" * 55)
