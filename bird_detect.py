import os
import datetime
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLO model for bird detection
bird_model = YOLO("yolov8n.pt")  

# Create directories to store images
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

@app.route("/")
def home():
    return "Bird Detection Backend Running!"

def process_image(image_path):
    """Process an image using YOLO and return detections."""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Invalid image"}

    results = bird_model.predict(source=[image], conf=0.1, save=False)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            bb = box.xyxy.numpy()[0]
            conf = box.conf.numpy()[0]
            class_id = int(box.cls.numpy()[0])
            class_name = bird_model.names[class_id]

            print(f"Detected: {class_name} | Confidence: {conf:.2f}")

            cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 3)
            cv2.putText(image, f"{class_name} {conf:.2f}", (int(bb[0]), int(bb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            detected_objects.append({
                "class": class_name,
                "confidence": float(conf),
                "bounding_box": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
            })

    processed_filename = "processed_" + os.path.basename(image_path)
    processed_path = os.path.join("processed", processed_filename)
    cv2.imwrite(processed_path, image)

    return {"detections": detected_objects, "result_image": f"/processed/{processed_filename}"}

@app.route("/detect/bird", methods=["POST"])
def detect_bird():
    """Detect birds in an uploaded image."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = datetime.datetime.now().strftime("bird_%Y-%m-%d_%H-%M-%S.jpg")
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    results = process_image(filepath)
    return jsonify(results)

@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_from_directory("processed", filename)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
