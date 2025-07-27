from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import os
import pytesseract
import re

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


MODEL_PATH = "weather_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


def classify_weather(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  
    img = np.expand_dims(img, axis=0) / 255.0  
    prediction = model.predict(img)
    classes = ["sunny", "night", "cloudy", "other"]
    return classes[np.argmax(prediction)]


def extract_coordinates(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    
    lat_long_match = re.search(r'Lat(?:itude)?[:\s]*([-\d.]+).*?Long(?:itude)?[:\s]*([-\d.]+)', text, re.IGNORECASE)
    
    if lat_long_match:
        try:
            latitude = float(lat_long_match.group(1))
            longitude = float(lat_long_match.group(2))
            if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                return latitude, longitude
        except:
            pass
    return None, None


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    
    weather_condition = classify_weather(filepath)

    
    latitude, longitude = extract_coordinates(filepath)

    return jsonify({
        "weather_condition": weather_condition,
        "latitude": latitude,
        "longitude": longitude,
        "image_url": f"http://127.0.0.1:5000/uploads/{file.filename}"
    })


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
