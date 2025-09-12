from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import gdown
import requests
from ultralytics import YOLO
import logging
import re

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
class_names = []

def extract_file_id_from_url(url):
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'/d/([a-zA-Z0-9-_]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_model_from_gdrive():
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        logger.info(f"✅ Model already exists at {model_path}")
        return True

    try:
        os.makedirs('models', exist_ok=True)
        gdrive_input = os.getenv('GDRIVE_FILE_ID')
        if not gdrive_input:
            logger.error("❌ GDRIVE_FILE_ID environment variable not set")
            return False

        file_id = extract_file_id_from_url(gdrive_input) if gdrive_input.startswith("http") else gdrive_input
        if not file_id:
            logger.error(f"❌ Cannot extract file ID from {gdrive_input}")
            return False

        logger.info(f"⬇️ Downloading model from Google Drive (file_id={file_id})...")
        output = gdown.download(id=file_id, output=model_path, quiet=False, fuzzy=True)

        if output and os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            logger.info(f"✅ Model downloaded successfully! Size: {os.path.getsize(model_path)} bytes")
            return True
        else:
            logger.error("❌ gdown failed or file too small.")
            if os.path.exists(model_path):
                os.remove(model_path)
            return False

    except Exception as e:
        logger.error(f"❌ Error downloading model from Google Drive: {str(e)}")
        return False

def is_valid_pytorch_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(10)
        if header.startswith(b'<') or header.startswith(b'<!'):
            logger.error("File appears to be HTML, not a PyTorch model")
            return False
        file_size = os.path.getsize(file_path)
        if file_size < 1000000:
            logger.warning(f"File size ({file_size} bytes) seems too small for a YOLO model")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating model file: {str(e)}")
        return False

def download_model_from_url():
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return True

    try:
        os.makedirs('models', exist_ok=True)
        model_url = os.getenv('MODEL_URL')
        if not model_url:
            logger.info("MODEL_URL environment variable not set, skipping URL download")
            return False

        logger.info(f"Downloading model from URL: {model_url}")

        if 'drive.google.com' in model_url:
            file_id = extract_file_id_from_url(model_url)
            if file_id:
                model_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"
                logger.info(f"Converted to direct download URL: {model_url}")

        response = requests.get(model_url, stream=True, allow_redirects=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if os.path.exists(model_path) and os.path.getsize(model_path) > 100000 and is_valid_pytorch_model(model_path):
            logger.info("Model downloaded successfully from URL!")
            return True
        else:
            logger.error("Downloaded file is invalid or too small")
            if os.path.exists(model_path):
                os.remove(model_path)
            return False
    except Exception as e:
        logger.error(f"Error downloading model from URL: {str(e)}")
        return False

def load_model():
    global model, class_names
    try:
        model_path = 'models/best.pt'
        gdrive_id = os.getenv('GDRIVE_FILE_ID', 'NOT_SET')
        model_url = os.getenv('MODEL_URL', 'NOT_SET')
        logger.info(f"GDRIVE_FILE_ID: {'SET' if gdrive_id != 'NOT_SET' else 'NOT_SET'}")
        logger.info(f"MODEL_URL: {'SET' if model_url != 'NOT_SET' else 'NOT_SET'}")

        if not os.path.exists(model_path):
            logger.info("Model file not found, attempting download...")
            if not download_model_from_gdrive():
                logger.info("Google Drive download failed, trying URL method...")
                if not download_model_from_url():
                    logger.error("Failed to download model from all sources")
                    return False

        if not os.path.exists(model_path):
            logger.error(f"Model file still not found at {model_path}")
            return False

        if not is_valid_pytorch_model(model_path):
            logger.error("Model file validation failed")
            return False

        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        class_names = model.names
        logger.info(f"Model loaded successfully. Classes: {class_names}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def process_image(image_data):
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(cv_image)

        detections = []
        annotated_image = cv_image.copy()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
                    if confidence > confidence_threshold:
                        class_name = class_names.get(class_id, f"Class_{class_id}")
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_image, label, (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'success': True,
            'detections': detections,
            'annotated_image': f"data:image/jpeg;base64,{annotated_b64}",
            'total_detections': len(detections)
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {'success': False, 'error': str(e)}

# ---------------- Flask Routes ----------------

@app.route('/')
def index():
    return render_template('index.html')  # ต้องมีไฟล์ templates/index.html

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'})
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded yet'})
    result = process_image(data['image'])
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': class_names if class_names else {},
        'model_path': 'models/best.pt' if os.path.exists('models/best.pt') else None
    })

@app.route('/download-model', methods=['POST'])
def force_download_model():
    global model, class_names
    if os.path.exists('models/best.pt'):
        os.remove('models/best.pt')
    model = None
    class_names = []
    if load_model():
        return jsonify({'success': True, 'message': 'Model downloaded and loaded successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to download model'})

# ---------------- Run App ----------------

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    if load_model():
        logger.info("Model loaded successfully. Starting server...")
    else:
        logger.warning("Model loading failed. Server will start but predictions won't work until model is loaded.")
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
