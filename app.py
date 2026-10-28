
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"
model = joblib.load(MODEL_PATH)

EXPECTED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "TotalChargesPerTenure",
    "ServiceCount",
    "HasMultipleServices",
]


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object containing customer details."
        }), 400

    missing = [col for col in EXPECTED_COLUMNS if col not in data]
    if missing:
        return jsonify({
            "error": "Required fields are missing.",
            "missing_fields": missing
        }), 400

    try:
        customer = {col: data[col] for col in EXPECTED_COLUMNS}
        input_df = pd.DataFrame([customer], columns=EXPECTED_COLUMNS)

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        return jsonify({
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "churn_probability_percent": round(probability * 100, 2)
        }), 200

    except (ValueError, TypeError) as exc:
        return jsonify({
            "error": "Invalid customer input.",
            "details": str(exc)
        }), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
