"""
app.py  —  P653 Diabetic Retinopathy Prediction Platform
=========================================================
Clean Flask backend: loads pre-trained model, serves existing UI,
and exposes prediction API endpoints.

Directory layout expected:
    .
    ├── app.py                  ← this file
    ├── templates/
    │   └── index.html          ← existing frontend (untouched)
    └── models/
        ├── dp1_model.pkl       ← trained LogisticRegression
        ├── dp1_scaler.pkl      ← fitted StandardScaler
        └── dp1_features.pkl    ← feature name list
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ── Silence sklearn version warnings in production ─────────────────────────────
warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# APP FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    CORS(app)                         # allow cross-origin requests from UI
    _register_routes(app)
    return app


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER  (singleton — loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "dp1_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "dp1_scaler.pkl")
FEAT_PATH = os.path.join(MODELS_DIR, "dp1_features.pkl")

# Canonical feature order (must match training notebook cell 25)
ALL_FEATURES: list[str] = [
    "age", "systolic_bp", "diastolic_bp", "cholesterol",
    "pulse_pressure", "bp_ratio", "age_systolic", "chol_age_ratio",
    "hypertension_score", "high_cholesterol", "age_group_enc",
]

_model = None
_scaler = None
_model_ready = False


def load_model() -> None:
    """Load model artefacts from disk into module-level singletons."""
    global _model, _scaler, _model_ready
    try:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _model_ready = True
        logger.info("✅  Model loaded  →  %s", MODEL_PATH)
    except FileNotFoundError as exc:
        logger.error("❌  Model file not found: %s", exc)
        logger.error(
            "    Train first:  python train.py   (or run app_dp1.py --train)")


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# Hard physiological limits used for input validation
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age":          (18,  110),
    "systolic_bp":  (60,  250),
    "diastolic_bp": (30,  160),
    "cholesterol":  (50,  400),
}

# Reference "normal" ranges used for abnormality flags in the response
NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "systolic_bp":  (0,   120),
    "diastolic_bp": (0,    80),
    "cholesterol":  (125, 200),
}

RISK_TIERS = [
    {"tier": "MINIMAL",  "label": "Minimal Risk",
        "range": "0–25%",   "consult": False, "max": 0.25},
    {"tier": "LOW",      "label": "Low Risk",
        "range": "25–55%",  "consult": False, "max": 0.55},
    {"tier": "MODERATE", "label": "Moderate Risk",
        "range": "55–80%",  "consult": True,  "max": 0.80},
    {"tier": "HIGH",     "label": "High Risk",
        "range": "80–100%", "consult": True,  "max": 1.01},
]


# ══════════════════════════════════════════════════════════════════════════════
# PURE HELPER FUNCTIONS  (stateless, easily unit-tested)
# ══════════════════════════════════════════════════════════════════════════════
def _validate_inputs(
    age: float, sbp: float, dbp: float, chol: float
) -> dict[str, str]:
    """Return a dict of {field: warning_message} for out-of-range values."""
    values = {
        "age": age, "systolic_bp": sbp,
        "diastolic_bp": dbp, "cholesterol": chol,
    }
    warnings_out: dict[str, str] = {}
    for field, val in values.items():
        lo, hi = FEATURE_RANGES[field]
        if not (lo <= val <= hi):
            warnings_out[field] = (
                f"Value {val} is outside the expected physiological range [{lo}, {hi}]"
            )
    return warnings_out


def _abnormal_flags(sbp: float, dbp: float, chol: float) -> dict[str, bool]:
    """Flag readings outside the reference normal range."""
    return {
        "systolic_bp":  sbp > NORMAL_RANGES["systolic_bp"][1],
        "diastolic_bp": dbp > NORMAL_RANGES["diastolic_bp"][1],
        "cholesterol": not (NORMAL_RANGES["cholesterol"][0]
                            <= chol
                            <= NORMAL_RANGES["cholesterol"][1]),
    }


def _risk_tier(prob: float) -> dict:
    """Map a probability to the matching risk tier descriptor."""
    for tier in RISK_TIERS:
        if prob < tier["max"]:
            return tier
    return RISK_TIERS[-1]


def _engineer_features(
    age: float, sbp: float, dbp: float, chol: float
) -> np.ndarray:
    """
    Reproduce the feature-engineering pipeline from notebook cell 25.
    Returns a (1, 11) float64 array ready for the scaler.
    """
    pulse_pressure = sbp - dbp
    bp_ratio = sbp / max(dbp, 1e-6)
    age_systolic = age * sbp
    chol_age_ratio = chol / max(age, 1e-6)
    hypertension_score = int(sbp > 130) + int(dbp > 80)
    high_cholesterol = int(chol > 200)
    age_group_enc = (
        0 if age < 40 else
        1 if age < 55 else
        2 if age < 65 else
        3
    )
    return np.array(
        [age, sbp, dbp, chol,
         pulse_pressure, bp_ratio, age_systolic, chol_age_ratio,
         hypertension_score, high_cholesterol, age_group_enc],
        dtype=np.float64,
    ).reshape(1, -1)


def _statistical_fallback(
    age: float, sbp: float, dbp: float, chol: float
) -> float:
    """
    Lightweight logistic approximation used when model artefacts are absent.
    Coefficients mirror the synthetic training distribution in the notebook.
    """
    logit = (
        0.030 * (age - 64)
        + 0.025 * (sbp - 105)
        + 0.020 * (dbp - 92)
        + 0.022 * (chol - 110)
    )
    pp = sbp - dbp
    bpr = sbp / max(dbp, 1)
    cv = (age * sbp * chol) / 1e6
    logit += (
        0.018 * (pp - 13) / 15
        + 0.014 * (bpr - 1.15) / 0.25
        + 0.016 * (cv - 0.8) / 0.5
        + 0.05
    )
    return float(1.0 / (1.0 + np.exp(-logit)))


def run_inference(age: float, sbp: float, dbp: float, chol: float) -> dict:
    """
    Core inference function.
    Uses the loaded sklearn model when available, falls back to a
    statistical approximation otherwise.

    Returns a structured response dict consumed directly by the frontend.
    """
    feat_vector = _engineer_features(age, sbp, dbp, chol)

    if _model_ready and _model is not None and _scaler is not None:
        scaled = _scaler.transform(feat_vector)
        probability = float(_model.predict_proba(scaled)[0][1])
        prediction = int(_model.predict(scaled)[0])
        source = "scikit-learn (GridSearchCV — LogisticRegression)"
    else:
        probability = _statistical_fallback(age, sbp, dbp, chol)
        prediction = int(probability >= 0.5)
        source = "statistical_fallback — model artefacts not found"

    tier = _risk_tier(probability)

    # Derived values surfaced in the response for UI display
    pp = float(sbp - dbp)
    bpr = float(sbp / max(dbp, 1))
    asx = float(age * sbp)
    car = float(chol / max(age, 1))
    cv = float((age * sbp * chol) / 1e6)
    map_ = float((sbp + 2 * dbp) / 3)
    hs = int(sbp > 130) + int(dbp > 80)
    hc = int(chol > 200)
    ag = 0 if age < 40 else (1 if age < 55 else (2 if age < 65 else 3))

    return {
        # ── Primary result ──────────────────────────────────────────────────
        "prediction":        "retinopathy" if prediction else "no_retinopathy",
        "prediction_binary": prediction,
        "detected":          bool(prediction),
        "probability":       round(probability, 4),
        "probability_pct":   round(probability * 100, 1),
        # ── Risk classification ─────────────────────────────────────────────
        "risk_tier":         tier["tier"],
        "risk_tier_label":   tier["label"],
        "risk_range":        tier["range"],
        "consult_doctor":    tier["consult"],
        # ── Audit / transparency ────────────────────────────────────────────
        "inference_source":  source,
        "features_computed": len(ALL_FEATURES),
        # ── Echo inputs ─────────────────────────────────────────────────────
        "inputs": {
            "age":          age,
            "systolic_bp":  sbp,
            "diastolic_bp": dbp,
            "cholesterol":  chol,
        },
        # ── Derived features (for UI detail panel) ──────────────────────────
        "derived": {
            "pulse_pressure":         round(pp,  4),
            "bp_ratio":               round(bpr, 4),
            "age_systolic":           round(asx, 2),
            "chol_age_ratio":         round(car, 4),
            "hypertension_score":     hs,
            "high_cholesterol":       hc,
            "age_group_enc":          ag,
            "cv_risk_score":          round(cv,   4),
            "mean_arterial_pressure": round(map_, 2),
        },
        # ── Clinical flags ──────────────────────────────────────────────────
        "validation_warnings": _validate_inputs(age, sbp, dbp, chol),
        "abnormal_flags":      _abnormal_flags(sbp, dbp, chol),
        # ── Metadata ────────────────────────────────────────────────────────
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
def _register_routes(app: Flask) -> None:

    # ── UI ────────────────────────────────────────────────────────────────────
    @app.get("/")
    def index():
        """Serve the existing frontend from templates/index.html."""
        return render_template("index.html")

    # ── Single prediction ─────────────────────────────────────────────────────
    @app.post("/predict")
    def predict():
        """
        Accept JSON body:
            { "age": 65, "systolic_bp": 140, "diastolic_bp": 90, "cholesterol": 220 }

        Returns full inference result as JSON.
        """
        payload = request.get_json(force=True, silent=True) or {}

        # Parse & type-coerce required fields
        try:
            age = float(payload["age"])
            sbp = float(payload["systolic_bp"])
            dbp = float(payload["diastolic_bp"])
            chol = float(payload["cholesterol"])
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": f"Missing or invalid field — {exc}"}), 400

        # Reject completely out-of-bounds values
        warnings_found = _validate_inputs(age, sbp, dbp, chol)
        missing = {k: v for k, v in warnings_found.items()
                   if "outside" not in v}
        if missing:
            return jsonify({"error": f"Invalid input: {missing}"}), 400

        result = run_inference(age, sbp, dbp, chol)
        return jsonify(result), 200

    # ── Batch prediction ──────────────────────────────────────────────────────
    @app.post("/batch")
    def batch_predict():
        """
        Accept JSON body:
            { "records": [ { "age": ..., "systolic_bp": ..., ... }, ... ] }

        Returns aggregated results with per-record predictions.
        """
        payload = request.get_json(force=True, silent=True) or {}
        records = payload.get("records", [])

        if not isinstance(records, list) or len(records) == 0:
            return jsonify({"error": "'records' must be a non-empty list"}), 400

        results: list[dict] = []
        errors:  list[dict] = []

        for idx, rec in enumerate(records):
            try:
                results.append(run_inference(
                    float(rec["age"]),
                    float(rec["systolic_bp"]),
                    float(rec["diastolic_bp"]),
                    float(rec["cholesterol"]),
                ))
            except Exception as exc:
                errors.append({"index": idx, "error": str(exc)})

        return jsonify({
            "count":              len(results),
            "retinopathy_count":  sum(1 for r in results if r["detected"]),
            "results":            results,
            "errors":             errors,
        }), 200

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        """Liveness + readiness probe used by the UI model-status badge."""
        return jsonify({
            "status":         "ok",
            "model_loaded":   _model_ready,
            "model_path":     MODEL_PATH,
            "scaler_path":    SCALER_PATH,
            "inference_mode": "scikit-learn" if _model_ready else "statistical_fallback",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }), 200

    # ── Model metadata ────────────────────────────────────────────────────────
    @app.get("/model/info")
    def model_info():
        """Returns model metadata: algorithm, features, risk tier definitions."""
        info: dict = {
            "project":       "P653 — Diabetic Retinopathy",
            "algorithm":     "Logistic Regression (GridSearchCV)",
            "features":      ALL_FEATURES,
            "feature_count": len(ALL_FEATURES),
            "model_loaded":  _model_ready,
            "risk_tiers": [
                {k: v for k, v in t.items() if k != "max"}
                for t in RISK_TIERS
            ],
        }
        if _model_ready and _model is not None:
            try:
                info["model_params"] = _model.get_params()
            except Exception:
                pass
        return jsonify(info), 200

    # ── Feature schema ────────────────────────────────────────────────────────
    @app.get("/features")
    def features_schema():
        """Documents the input contract and derived feature pipeline."""
        return jsonify({
            "input": [
                {"name": "age",          "unit": "years", "range": [18, 110]},
                {"name": "systolic_bp",  "unit": "mmHg",
                    "range": [60, 250], "normal": "<120"},
                {"name": "diastolic_bp", "unit": "mmHg",
                    "range": [30, 160], "normal": "<80"},
                {"name": "cholesterol",  "unit": "mg/dL",
                    "range": [50, 400], "normal": "125–200"},
            ],
            "derived": [
                {"name": "pulse_pressure",
                    "formula": "systolic_bp - diastolic_bp"},
                {"name": "bp_ratio",
                    "formula": "systolic_bp / diastolic_bp"},
                {"name": "age_systolic",            "formula": "age × systolic_bp"},
                {"name": "chol_age_ratio",          "formula": "cholesterol / age"},
                {"name": "hypertension_score",
                    "formula": "int(sbp>130) + int(dbp>80)"},
                {"name": "high_cholesterol",
                    "formula": "int(cholesterol > 200)"},
                {"name": "age_group_enc",
                    "formula": "LabelEncoder → <40/40-55/55-65/65+"},
                {"name": "cv_risk_score",
                    "formula": "(age × sbp × chol) / 1e6"},
                {"name": "mean_arterial_pressure",
                    "formula": "(sbp + 2×dbp) / 3"},
            ],
            "output": {
                "prediction":     "retinopathy | no_retinopathy",
                "probability":    "float [0.0, 1.0]",
                "risk_tier":      "MINIMAL | LOW | MODERATE | HIGH",
                "consult_doctor": "bool",
            },
        }), 200

    # ── Error handlers ────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def server_error(exc):
        logger.exception("Unhandled exception")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
app = create_app()

if __name__ == "__main__":
    load_model()

    port = int(os.environ.get("PORT", 5000))
    # off by default in production
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    logger.info("🚀  P653 RetinaSight  →  http://0.0.0.0:%d", port)
    logger.info("    Model ready : %s", _model_ready)
    logger.info("    Debug mode  : %s", debug)

    app.run(host="0.0.0.0", port=port, debug=debug)
