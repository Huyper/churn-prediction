from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

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
