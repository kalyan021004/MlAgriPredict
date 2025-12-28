from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "crop_recommend_svm_soil.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "crop_scaler_soil.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder_soil.pkl"))
soil_encoder = joblib.load(os.path.join(MODEL_DIR, "soil_encoder.pkl"))

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/api/ml/crop-recommendation", methods=["POST"])
def crop_recommendation():
    data = request.json or {}

    N = float(data.get("N", 60))
    P = float(data.get("P", 40))
    K = float(data.get("K", 40))
    temperature = float(data.get("temperature", 25))
    humidity = float(data.get("humidity", 70))
    ph = float(data.get("ph", 6.5))
    rainfall = float(data.get("rainfall", 600))
    soil = data.get("soil", "loamy").lower()

    soil_encoded = soil_encoder.transform([soil])[0]

    X = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_encoded]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    crop = label_encoder.inverse_transform([pred])[0]

    return jsonify({
        "crop": crop,
        "confidence": 0.75,
        "expectedYield": "Model-based estimate",
        "reasoning": "ML prediction using soil + nutrient + climate features"
    })

if __name__ == "__main__":
    app.run()
