"""
wsgi.py  —  Production entry point for Gunicorn
================================================
Downloads model artefacts from Google Drive on first startup,
then loads them into the Flask app.
"""

from app import app, load_model
import os
import gdown

os.makedirs("models", exist_ok=True)

# ── Download model files from Google Drive (only if not already present) ───────
if not os.path.exists("models/dp1_model.pkl"):
    print("⬇️  Downloading model files from Google Drive...")

    gdown.download(
        id="1R_NR9-jTV0dgnD7vB-P8l2TUPnGIPISo",
        output="models/dp1_model.pkl",
        quiet=False
    )
    gdown.download(
        id="1fNe2CDipKGji1Wnp3w_UhokqcGMF83Px",
        output="models/dp1_scaler.pkl",
        quiet=False
    )
    gdown.download(
        id="1-1CITiTppyvyZtSx7IxWZnaqtQhockeO",
        output="models/dp1_features.pkl",
        quiet=False
    )
    print("✅  Model files downloaded successfully.")
else:
    print("✅  Model files already present — skipping download.")

# ── Load Flask app and model ───────────────────────────────────────────────────

load_model()

application = app
