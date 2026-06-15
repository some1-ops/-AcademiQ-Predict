# AcademiQ Predict — Student Performance Prediction System

A fully **offline**, **self-contained** AI-powered desktop application that uses a **Decision Tree (J48/C4.5 — Entropy criterion)** to classify student academic performance into five categories: `Excellent`, `Good`, `Average`, `Poor`, and `Fail`.

---

## 🚀 Quick Start

### Option A — Double-click launcher (Windows)
```
Double-click  run.bat
```
The launcher will install all dependencies and open the app at **http://localhost:8501** automatically.

### Option B — Terminal
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 🔐 Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Administrator | `admin` | `admin123` |
| Student | `student` | `student123` |

> Credentials are stored in a local SQLite database (`data/students.db`). The database is created automatically on first launch.

---

## 📋 System Requirements

- **Python:** 3.9 – 3.12 recommended (3.14 works with protobuf 3.20.3 downgrade applied automatically)
- **OS:** Windows / macOS / Linux
- **Internet:** Not required after install — 100% offline operation

---

## 📦 Dependencies

Installed automatically via `run.bat` or `pip install -r requirements.txt`:

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI (runs locally on localhost) |
| `scikit-learn` | Decision Tree classifier (J48/C4.5 equivalent) |
| `pandas` | Dataset loading and preprocessing |
| `numpy` | Numerical operations |
| `matplotlib` | Tree visualization and charts |
| `seaborn` | Confusion matrix heatmap |
| `joblib` | Model persistence (save/load) |
| `openpyxl` | Excel file support |

---

## 📁 Project Structure

```
prediction system with weka/
├── app.py                      # Main entry point
├── run.bat                     # One-click Windows launcher
├── requirements.txt
│
├── core/
│   ├── auth.py                 # Login / role management
│   ├── database.py             # SQLite schema and CRUD
│   ├── ml_engine.py            # Decision Tree training & prediction
│   ├── data_validator.py       # CSV/Excel schema validation
│   └── mock_data.py            # 200-row synthetic dataset generator
│
├── pages/
│   ├── p1_dashboard.py         # Dashboard / home
│   ├── p2_dataset_upload.py    # File upload + validation
│   ├── p3_model_training.py    # Train model + view metrics
│   ├── p4_single_prediction.py # Single student prediction form
│   └── p5_batch_prediction.py  # Batch CSV prediction + export
│
└── data/                       # Auto-created at runtime
    ├── students.db             # SQLite database
    ├── model.pkl               # Saved trained model
    ├── mock_data.csv           # Auto-generated seed data
    └── uploads/                # Uploaded datasets
```

---

## 🗂️ Input Data Schema

Your CSV/Excel dataset must include these columns:

| Column | Type | Range |
|--------|------|-------|
| Student ID | String | Unique identifier |
| Attendance | Float | 0 – 100 (%) |
| Assignment Score | Float | 0 – 100 |
| Test Score | Float | 0 – 100 |
| Study Hours | Float | 0 – 168 (weekly) |
| Class Participation | Integer | 1 – 5 |
| Previous GPA | Float | 0.00 – 4.00 |
| Performance Class | Categorical | Excellent / Good / Average / Poor / Fail |

---

## 🧠 Machine Learning Model

- **Algorithm:** `DecisionTreeClassifier` with `criterion='entropy'`
- **Equivalent to:** WEKA J48 / C4.5 algorithm
- **Split:** 80% training / 20% testing (stratified)
- **Persistence:** Model saved as `data/model.pkl` via `joblib`
- **Auto-seed:** 200-row synthetic dataset generated on first launch if no data exists

---

## 🖥️ Pages & Features

| Page | Role | Description |
|------|------|-------------|
| Dashboard | All | System status, metrics, quick-start guide |
| Dataset Upload | Admin | Upload CSV/Excel, validate schema, preview data |
| Model Training | Admin | Train model, view accuracy/F1/confusion matrix, visualize tree |
| Single Prediction | All | Enter student profile → get instant classification |
| Batch Prediction | Admin | Upload unlabelled CSV → predict all → download results |

---

## 🔒 Security Notes

- All data is stored **locally only** — no network calls, no cloud
- Passwords are hashed with SHA-256 before storage
- Default credentials should be changed for production use via the SQLite database

---

## 📊 Expected Performance (on mock dataset)

| Metric | Typical Value |
|--------|--------------|
| Accuracy | 75 – 95% |
| F1-Score (macro) | 74 – 93% |
| Tree Depth | 5 – 15 levels |
