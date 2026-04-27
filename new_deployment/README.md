# P653 — Diabetic Retinopathy Prediction Platform

Clean, production-ready Flask backend that connects the pre-trained model
to the existing frontend UI. No model training code. No UI code in Python.

---

## Project Structure

```
P653-RetinaSight/
├── app.py                  ← Flask backend  (this is the only file you need to run)
├── wsgi.py                 ← Gunicorn entry point for production
├── requirements.txt
├── templates/
│   └── index.html          ← Existing frontend UI  (untouched)
└── models/
    ├── dp1_model.pkl       ← Trained LogisticRegression  (from notebook)
    ├── dp1_scaler.pkl      ← Fitted StandardScaler
    └── dp1_features.pkl    ← Feature name list
```

---

## Setup

```bash
# 1. Create & activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your trained model files
#    Copy dp1_model.pkl, dp1_scaler.pkl, dp1_features.pkl  →  models/
```

---

## Run

### Development
```bash
python app.py
# → http://localhost:5000
```

### Production (Gunicorn)
```bash
gunicorn wsgi:application --workers 4 --bind 0.0.0.0:5000
```

---

## API Reference

| Method | Endpoint       | Description                        |
|--------|----------------|------------------------------------|
| GET    | `/`            | Serve the frontend UI              |
| POST   | `/predict`     | Single-record prediction           |
| POST   | `/batch`       | Batch prediction                   |
| GET    | `/health`      | Liveness + model readiness probe   |
| GET    | `/model/info`  | Model metadata & risk tier config  |
| GET    | `/features`    | Input/output feature schema        |

### POST /predict — Request
```json
{
  "age":          65,
  "systolic_bp":  145,
  "diastolic_bp": 92,
  "cholesterol":  230
}
```

### POST /predict — Response
```json
{
  "prediction":        "retinopathy",
  "prediction_binary": 1,
  "detected":          true,
  "probability":       0.7821,
  "probability_pct":   78.2,
  "risk_tier":         "MODERATE",
  "risk_tier_label":   "Moderate Risk",
  "risk_range":        "55–80%",
  "consult_doctor":    true,
  "inference_source":  "scikit-learn (GridSearchCV — LogisticRegression)",
  "inputs":   { "age": 65, "systolic_bp": 145, "diastolic_bp": 92, "cholesterol": 230 },
  "derived":  { "pulse_pressure": 53.0, "bp_ratio": 1.576, ... },
  "validation_warnings": {},
  "abnormal_flags":  { "systolic_bp": true, "diastolic_bp": true, "cholesterol": true },
  "timestamp": "2025-01-01T12:00:00+00:00"
}
```

### POST /batch — Request
```json
{
  "records": [
    { "age": 65, "systolic_bp": 145, "diastolic_bp": 92, "cholesterol": 230 },
    { "age": 42, "systolic_bp": 118, "diastolic_bp": 76, "cholesterol": 185 }
  ]
}
```

---

## Environment Variables

| Variable       | Default | Description                       |
|----------------|---------|-----------------------------------|
| `PORT`         | `5000`  | HTTP port                         |
| `FLASK_DEBUG`  | `0`     | Set to `1` for hot-reload in dev  |

---

## Fallback Mode

If model `.pkl` files are missing, the backend automatically falls back to
a statistical approximation (same logit coefficients used to synthesise the
training data). The `/health` endpoint reports `"model_loaded": false` and
the UI displays a warning badge.

Train and export the model from the notebook, then copy the three `.pkl`
files into `models/` to activate full ML inference.
