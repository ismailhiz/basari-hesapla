from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Modeli yükle
model = joblib.load("linear_regression_model.pkl")

@app.route("/")
def home():
    return "Yapay zeka modeli yayında! POST /predict"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Railway PORT kullanır
    app.run(host='0.0.0.0', port=port)
