import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import shap
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# ─────────────────────────────────────────────
# KONFIGURASI KEAMANAN (API KEY)
# ─────────────────────────────────────────────
# Di dunia nyata, simpan ini di file .env atau di environment variable
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    secret_api_key = os.getenv("SECRET_API_KEY")
    if not secret_api_key:
        raise HTTPException(status_code=500, detail="Server configuration error: SECRET_API_KEY tidak diset")
    if api_key_header == secret_api_key:
        return api_key_header
    raise HTTPException(status_code=403, detail="Akses Ditolak: API Key tidak valid!")

# ─────────────────────────────────────────────
# INISIALISASI FASTAPI & MUAT ARTEFAK
# ─────────────────────────────────────────────
app = FastAPI(
    title="TrustChain Fraud Detection API",
    description="API Pendeteksi Penipuan dengan Explainable AI (SHAP)",
    version="1.0.0"
)

# Variabel Global untuk Model
model = None
scaler = None
num_imputer = None
encoders = {}
iso_forest = None
feature_names = []
explainer = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, num_imputer, encoders, iso_forest, feature_names, explainer
    
    print("Memuat model dan artefak...")
    BASE_DIR = "artifacts"
    
    # Muat Model LSTM (compile=False agar cepat & aman)
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "best_lstm.keras"), 
        compile=False,
        custom_objects={"quantization_config": None}
    )
    
    # Muat Pipeline Preprocessing
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    num_imputer = joblib.load(os.path.join(BASE_DIR, "num_imputer.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))
    iso_forest = joblib.load(os.path.join(BASE_DIR, "isolation_forest.pkl"))
    
    # Muat Nama Fitur (Tambahkan Anomaly Score)
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    feature_names.append("Anomaly_Score_IF")
    
    # Setup SHAP Explainer
    print("Menyiapkan SHAP Explainer...")
    bg_data_2d = np.load(os.path.join(BASE_DIR, "background_sample_2d.npy"))
    
    # Fungsi wrapper untuk SHAP
    def predict_fn(X_2d):
        X_3d = X_2d.reshape(X_2d.shape[0], 1, X_2d.shape[1])
        preds = model.predict_on_batch(X_3d)
        return np.array(preds).ravel()
    
    # Rangkum background data agar komputasi API tidak memakan waktu lama
    bg_summary = shap.kmeans(bg_data_2d, 10)
    explainer = shap.KernelExplainer(predict_fn, bg_summary)
    print("Sistem siap menerima request!")

# ─────────────────────────────────────────────
# SKEMA REQUEST DARI USER (JSON)
# ─────────────────────────────────────────────
class TransactionRequest(BaseModel):
    data: list[dict]

# ─────────────────────────────────────────────
# ENDPOINT UTAMA
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "Online", "message": "TrustChain API is running. Gunakan endpoint /predict"}

@app.post("/predict")
def predict_fraud(request: TransactionRequest, api_key: str = Depends(get_api_key)):
    try:
        # 1. Konversi JSON ke DataFrame
        df = pd.DataFrame(request.data)
        
        # Pisahkan kolom sesuai tipe (hindari kolom yang tidak dikenali)
        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        num_cols = [c for c in df.columns if df[c].dtype != "object"]
        
        # 2. Preprocessing Data Kategorik
        df[cat_cols] = df[cat_cols].fillna("missing")
        for col in cat_cols:
            if col in encoders:
                # Tangani nilai baru (Unseen label) yang tidak ada saat training
                known_classes = list(encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else "missing")
                df[col] = encoders[col].transform(df[col].astype(str))
                
        # 3. Preprocessing Data Numerik
        df[num_cols] = num_imputer.transform(df[num_cols])
        
        # 4. Dapatkan Anomaly Score dari Isolation Forest
        raw_scores = iso_forest.decision_function(df)
        scores_inv = -raw_scores
        # Simulasi min-max statis (idealnya disimpan ke artefak, ini aproksimasi aman)
        scores_norm = (scores_inv - (-0.5)) / (0.5 - (-0.5) + 1e-9) 
        scores_norm = np.clip(scores_norm, 0, 1)
        
        # 5. Scaling
        df_scaled = scaler.transform(df)
        
        # 6. Gabungkan Fitur (2D Matrix siap pakai)
        X_aug_2d = np.hstack([df_scaled, scores_norm.reshape(-1, 1)])
        
        # 7. Prediksi LSTM
        X_lstm_3d = X_aug_2d.reshape(X_aug_2d.shape[0], 1, X_aug_2d.shape[1])
        preds = model.predict_on_batch(X_lstm_3d)
        prob = float(preds[0][0])
        is_fraud = bool(prob > 0.5) # Threshold bisa kamu sesuaikan di sini
        
        # 8. Eksekusi SHAP (Explainability)
        shap_vals = explainer.shap_values(X_aug_2d[0:1]) # Proses 1 transaksi saja
        
        # Format penjelasan SHAP
        explanation = []
        for i, feat_name in enumerate(feature_names):
            contribution = float(shap_vals[0][i])
            if abs(contribution) > 0.001: # Abaikan fitur yang tidak berkontribusi
                # Ambil nilai input asli (sebelum scaling) jika tersedia di df
                original_value = request.data[0].get(feat_name, "N/A")
                if feat_name == "Anomaly_Score_IF":
                    original_value = round(float(scores_norm[0]), 3)
                
                explanation.append({
                    "feature": feat_name,
                    "original_value": original_value,
                    "contribution": round(contribution, 4)
                })
        
        # Urutkan berdasarkan fitur paling berpengaruh (absolut terbesar)
        explanation.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_reasons = explanation[:5] # Ambil 5 alasan utama saja
        
        # 9. Kembalikan Response
        return {
            "status": "success",
            "prediction": {
                "fraud_probability": round(prob, 4),
                "is_fraud": is_fraud,
                "confidence_percentage": f"{round(prob * 100, 2)}%" if is_fraud else f"{round((1 - prob) * 100, 2)}%"
            },
            "explainability": {
                "message": "Fitur-fitur ini sangat mendorong transaksi ke arah " + ("Fraud" if is_fraud else "Normal"),
                "top_influencers": top_reasons
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Terjadi kesalahan saat memproses data: {str(e)}")