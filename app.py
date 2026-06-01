# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import pickle

# app = Flask(__name__)
# CORS(app)

# # Load model + scaler
# model = tf.keras.models.load_model("IDS_ANN_Model.keras")

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         features = np.array(data["features"]).reshape(1, -1)

#         if features.shape[1] != 7:
#             return jsonify({"error": "Expected 7 features"}), 400

#         # Scale input
#         features_scaled = scaler.transform(features)

#         # Predict
#         prediction = model.predict(features_scaled)[0][0]
#         result = 1 if prediction >= 0.5 else 0

#         return jsonify({"prediction": result})

#     except Exception as e:
#         return jsonify({"error": str(e)})


# @app.route("/", methods=["GET"])
# def home():
#     return "IDS Backend Running Successfully!"


# if __name__ == "__main__":
#     app.run(debug=True)


import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and scaler ONCE (important for performance)
model = tf.keras.models.load_model("IDS_ANN_Model.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Home route
@app.route("/", methods=["GET"])
def home():
    return "Intrusion Detection System Backend Running Successfully!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None or "features" not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        # Convert input to numpy array
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)

        # Validate feature count
        if features.shape[1] != 7:
            return jsonify({"error": "Expected exactly 7 features"}), 400

        # Scale input
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)[0][0]

        # Binary output
        result = 1 if prediction >= 0.5 else 0

        return jsonify({
            "prediction": result,
            "confidence": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app (Gunicorn will handle this in production)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
