import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import shap
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import keras

@keras.saving.register_keras_serializable(package="Custom")
class SafeDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day"]  = (df["TransactionDT"] // 86400) % 7

    transaction_day = df["TransactionDT"] // 86400
    d_cols = [f"D{i}" for i in range(1, 16) if i != 7]

    for col in d_cols:
        norm_col = f"{col}_norm"
        df[norm_col] = (df[col] - transaction_day) if col in df.columns else np.nan

    df.drop(columns=[c for c in d_cols if c in df.columns], inplace=True)
    return df

# ─────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    secret = os.getenv("SECRET_API_KEY")
    if not secret:
        raise HTTPException(status_code=500, detail="SECRET_API_KEY tidak diset di server")
    if api_key == secret:
        return api_key
    raise HTTPException(status_code=403, detail="Akses Ditolak: API Key tidak valid!")

# ─────────────────────────────────────────────
# INISIALISASI FASTAPI & ARTEFAK
# ─────────────────────────────────────────────
app = FastAPI(
    title="TrustChain Fraud Detection API",
    description="API Pendeteksi Penipuan dengan Explainable AI (SHAP)",
    version="1.0.0"
)

model        = None
scaler       = None
num_imputer  = None
encoders     = {}
iso_forest   = None
feature_names = []
explainer    = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, num_imputer, encoders, iso_forest, feature_names, explainer

    logger.info("Memuat model dan artefak...")
    BASE_DIR = "artifacts"

    from keras.src.saving import serialization_lib
    _orig = serialization_lib.deserialize_keras_object

    def _patched(config, *args, **kwargs):
        if isinstance(config, dict):
            inner = config.get("config", {})
            if isinstance(inner, dict):
                inner.pop("quantization_config", None)
        return _orig(config, *args, **kwargs)

    serialization_lib.deserialize_keras_object = _patched

    model       = tf.keras.models.load_model(os.path.join(BASE_DIR, "best_lstm.keras"), compile=False)
    scaler      = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    num_imputer = joblib.load(os.path.join(BASE_DIR, "num_imputer.pkl"))
    encoders    = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))
    iso_forest  = joblib.load(os.path.join(BASE_DIR, "isolation_forest.pkl"))

    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    feature_names.append("Anomaly_Score_IF")

    logger.info("Menyiapkan SHAP Explainer...")
    bg_data_2d = np.load(os.path.join(BASE_DIR, "background_sample_2d.npy"))

    def predict_fn(X_2d):
        X_3d = X_2d.reshape(X_2d.shape[0], 1, X_2d.shape[1])
        return np.array(model.predict_on_batch(X_3d)).ravel()

    bg_summary = shap.kmeans(bg_data_2d, 10)
    explainer  = shap.KernelExplainer(predict_fn, bg_summary)
    logger.info("Sistem siap menerima request!")

# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────
class TransactionRequest(BaseModel):
    data: list[dict]

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "Online", "message": "TrustChain API is running. Gunakan endpoint /predict"}

@app.post("/predict")
def predict_fraud(request: TransactionRequest, api_key: str = Depends(get_api_key)):
    try:
        # ── 1. Konversi & Feature Engineering ──────────────────────────
        df = pd.DataFrame(request.data)
        logger.info(f"[1] Input shape: {df.shape}")

        df = apply_feature_engineering(df)
        logger.info(f"[2] Setelah FE: {df.shape}, kolom baru: {[c for c in df.columns if '_norm' in c or c in ('hour','day','TransactionAmt_log')]}")

        # ── 2. Pisahkan feature_names tanpa Anomaly_Score_IF ───────────
        feature_names_model = [f for f in feature_names if f != "Anomaly_Score_IF"]

        # ── 3. Encoding Kategorik ───────────────────────────────────────
        cat_cols = [c for c in feature_names_model if c in df.columns and df[c].dtype == "object"]
        df[cat_cols] = df[cat_cols].fillna("missing")
        for col in cat_cols:
            if col in encoders:
                known = list(encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "missing")
                df[col] = encoders[col].transform(df[col].astype(str))
            else:
                df[col] = 0
        logger.info(f"[3] Encoding kategorik selesai: {len(cat_cols)} kolom")

        # ── 4. Reindex sesuai urutan training ──────────────────────────
        df = df.reindex(columns=feature_names_model)

        # ── 5. Safety net: paksa sisa object → 0 ───────────────────────
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if obj_cols:
            logger.warning(f"[5] Kolom masih object setelah encode: {obj_cols} → diset 0")
            df[obj_cols] = 0

        # ── 6. Imputer untuk kolom numerik ─────────────────────────────
        num_cols_for_imputer = [c for c in feature_names_model if c not in encoders]
        df[num_cols_for_imputer] = num_imputer.transform(df[num_cols_for_imputer])
        logger.info(f"[6] Imputer selesai: {len(num_cols_for_imputer)} kolom numerik")

        # ── 7. Anomaly Score ────────────────────────────────────────────
        raw_scores  = iso_forest.decision_function(df)
        scores_norm = np.clip((-raw_scores + 0.5) / (1.0 + 1e-9), 0, 1)
        logger.info(f"[7] Anomaly score: {scores_norm[0]:.4f}")

        # ── 8. Scaling ──────────────────────────────────────────────────
        df_scaled = scaler.transform(df)
        logger.info(f"[8] Scaling selesai: shape {df_scaled.shape}")

        # ── 9. Gabungkan Anomaly Score ──────────────────────────────────
        X_aug_2d = np.hstack([df_scaled, scores_norm.reshape(-1, 1)])
        logger.info(f"[9] X_aug_2d shape: {X_aug_2d.shape}")

        # ── 10. Prediksi LSTM ───────────────────────────────────────────
        X_lstm_3d = X_aug_2d.reshape(X_aug_2d.shape[0], 1, X_aug_2d.shape[1])
        preds     = model.predict_on_batch(X_lstm_3d)
        prob      = float(preds[0][0])
        is_fraud  = prob > 0.5
        logger.info(f"[10] Prediksi: prob={prob:.4f}, is_fraud={is_fraud}")

        # ── 11. SHAP (opsional, tidak blokir response jika gagal) ───────
        top_influencers = []
        try:
            shap_vals   = explainer.shap_values(X_aug_2d[0:1])
            explanation = []
            for i, feat_name in enumerate(feature_names):
                contribution = float(shap_vals[0][i])
                if abs(contribution) > 0.001:
                    original_value = request.data[0].get(feat_name, "N/A")
                    if feat_name == "Anomaly_Score_IF":
                        original_value = round(float(scores_norm[0]), 3)
                    explanation.append({
                        "feature":        feat_name,
                        "original_value": original_value,
                        "contribution":   round(contribution, 4)
                    })
            explanation.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            top_influencers = explanation[:5]
            logger.info(f"[11] SHAP selesai: {len(top_influencers)} top features")
        except Exception as shap_err:
            logger.error(f"[11] SHAP gagal (diabaikan): {shap_err}")

        # ── 12. Response ────────────────────────────────────────────────
        return {
            "status": "success",
            "prediction": {
                "fraud_probability":     round(prob, 4),
                "is_fraud":              bool(is_fraud),
                "confidence_percentage": f"{round(prob * 100, 2)}%" if is_fraud else f"{round((1 - prob) * 100, 2)}%"
            },
            "explainability": {
                "message":         "Fitur-fitur ini sangat mendorong transaksi ke arah " + ("Fraud" if is_fraud else "Normal"),
                "top_influencers": top_influencers
            }
        }

    except Exception as e:
        logger.error(f"PREDICT ERROR: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Terjadi kesalahan: {str(e)}")