from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

CORS(app, origins="*", allow_headers=["Content-Type", "Authorization"])

model = tf.keras.models.load_model("waste_model.h5", compile=False)

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

IMG_SIZE = 224


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def home():
    return "Waste classifier backend is running"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "labels": labels
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin()
def predict():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response, 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files["image"]

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)