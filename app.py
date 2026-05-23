import cv2
import numpy as np
import torch
import base64
import pathlib
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# WINDOWS PATH FIX
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
CORS(app)

print("🔄 Model load ho raha hai... Thoda wait karein.")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
print("✅ Model load ho gaya!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image found"}), 400
            
        raw_image_data = data['image']
        
        if "," in raw_image_data:
            image_data = raw_image_data.split(",")[1]  # Strict cleaner string slice
        else:
            image_data = raw_image_data
            
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Decode failed"}), 400
            
        # YOLOv5 Inference
        results = model(frame)
        
        # 🔥 CRASH FIX: Direct python list format me parse karenge bina to_dict use kiye
        # coordinates structure: [xmin, ymin, xmax, ymax, confidence, class, name]
        raw_predictions = results.xyxy[0].cpu().numpy()
        names = results.names
        
        parsed_predictions = []
        for pred in raw_predictions:
            parsed_predictions.append({
                "xmin": float(pred[0]),
                "ymin": float(pred[1]),
                "xmax": float(pred[2]),
                "ymax": float(pred[3]),
                "confidence": float(pred[4]),
                "class": int(pred[5]),
                "name": names[int(pred[5])]
            })
        
        print(f"🎯 Successfully processed! Total boxes found: {len(parsed_predictions)}")
        return jsonify(parsed_predictions)
        
    except Exception as e:
        print(f"🚨 Python App Crash Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
