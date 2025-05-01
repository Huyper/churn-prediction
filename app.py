from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# Hàm tải file nếu chưa có
def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest} ...")
        r = requests.get(url)
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {dest}")

os.makedirs("models", exist_ok=True)

download_file("https://github.com/Huyper/churn-prediction/releases/download/v1.0.0/churn_data_dict.pkl", "models/churn_data_dict.pkl")
download_file("https://github.com/Huyper/churn-prediction/releases/download/v1.0.0/churn_encoder.pkl", "models/churn_encoder.pkl")
download_file("https://github.com/Huyper/churn-prediction/releases/download/v1.0.0/churn_model.pkl", "models/churn_model.pkl")

# Load model và các thành phần khác
model = joblib.load("models/churn_model.pkl")
encoder = joblib.load("models/churn_encoder.pkl")
data_dict = joblib.load("models/churn_data_dict.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Ví dụ: trích xuất features từ JSON đầu vào
        features = np.array([data[col] for col in data_dict["feature_order"]]).reshape(1, -1)
        features_encoded = encoder.transform(features)
        prediction = model.predict(features_encoded)
        return jsonify({"churn": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
