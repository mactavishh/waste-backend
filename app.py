from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Allow all origins (important for Flutter Web)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
model = tf.keras.models.load_model("waste_model.h5", compile=False)

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

# Test route
@app.route("/")
def home():
    return "Waste classifier backend is running"

# Prediction route
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    try:
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        index = int(np.argmax(predictions[0]))

        return jsonify({
            "class": labels[index],
            "confidence": float(predictions[0][index])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()