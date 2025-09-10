from flask import Flask, request, jsonify, render_template_string, render_template
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

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
class_names = []

def download_model_from_gdrive():
    """ดาวน์โหลด model จาก Google Drive"""
    model_path = 'models/best.pt'
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return True
        
    try:
        os.makedirs('models', exist_ok=True)
        logger.info("Downloading model from Google Drive...")
        
        # ใส่ Google Drive File ID ของคุณที่นี่
        file_id = os.getenv('GDRIVE_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID')
        
        if file_id == 'YOUR_GOOGLE_DRIVE_FILE_ID':
            logger.error("Please set GDRIVE_FILE_ID environment variable")
            return False
            
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # ใช้ gdown ดาวน์โหลด
        gdown.download(url, model_path, quiet=False)
        logger.info("Model downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model from Google Drive: {str(e)}")
        return False

def download_model_from_url():
    """ดาวน์โหลด model จาก URL"""
    model_path = 'models/best.pt'
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return True
        
    try:
        os.makedirs('models', exist_ok=True)
        
        # รับ URL จาก environment variable
        model_url = os.getenv('MODEL_URL')
        
        if not model_url:
            logger.error("MODEL_URL environment variable not set")
            return False
            
        logger.info(f"Downloading model from URL: {model_url}")
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='')
        
        print()  # New line
        logger.info("Model downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model from URL: {str(e)}")
        return False

def load_model():
    """Load YOLO model with auto-download"""
    global model, class_names
    
    try:
        model_path = 'models/best.pt'
        
        # ลองดาวน์โหลดจาก Google Drive ก่อน
        if not os.path.exists(model_path):
            if not download_model_from_gdrive():
                # ถ้าไม่ได้ ลองจาก URL
                if not download_model_from_url():
                    logger.error("Failed to download model from all sources")
                    return False
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        class_names = model.names
        logger.info(f"Model loaded successfully. Classes: {class_names}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def process_image(image_data):
    """Process image with YOLO model"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO prediction
        results = model(cv_image)
        
        # Process results
        detections = []
        annotated_image = cv_image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter by confidence threshold
                    confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
                    if confidence > confidence_threshold:
                        class_name = class_names.get(class_id, f"Class_{class_id}")
                        
                        # Add to detections list
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_image, label, (int(x1), int(y1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to base64
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

# HTML Template (same as before)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Object Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #007bff;
            background-color: #f8f9fa;
        }
        .upload-area.dragover {
            border-color: #007bff;
            background-color: #e3f2fd;
        }
        #fileInput {
            display: none;
        }
        .btn {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .results {
            margin-top: 30px;
        }
        .image-container {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .image-box {
            flex: 1;
            min-width: 300px;
        }
        .image-box img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .detection-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .detection-item {
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            border-left: 4px solid #007bff;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .model-status {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            text-align: center;
        }
        .model-loading {
            background: #fff3cd;
            color: #856404;
        }
        .model-ready {
            background: #d4edda;
            color: #155724;
        }
        .model-error {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 YOLO Object Detection</h1>
        
        <div id="modelStatus" class="model-status model-loading">
            ⏳ กำลังโหลด Model... กรุณารอสักครู่
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p>📁 คลิกหรือลากไฟล์รูปภาพมาที่นี่</p>
            <p>รองรับ: JPG, PNG, GIF</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*">
        <div style="text-align: center;">
            <button class="btn" onclick="processImage()" id="processBtn" disabled>🚀 วิเคราะห์รูปภาพ</button>
        </div>
        
        <div class="loading" id="loading">
            <p>⏳ กำลังประมวลผล...</p>
        </div>
        
        <div id="results" class="results" style="display: none;">
            <h3>📊 ผลการตรวจจับ</h3>
            <div id="detectionInfo" class="detection-info"></div>
            <div class="image-container">
                <div class="image-box">
                    <h4>รูปต้นฉบับ</h4>
                    <img id="originalImage" src="" alt="Original">
                </div>
                <div class="image-box">
                    <h4>ผลการตรวจจับ</h4>
                    <img id="annotatedImage" src="" alt="Detected">
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.querySelector('.upload-area');
        const processBtn = document.getElementById('processBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const modelStatus = document.getElementById('modelStatus');

        // Check model status on load
        checkModelStatus();

        function checkModelStatus() {
            fetch('/health')
                .then(response => response.json())
                .then(data => {
                    if (data.model_loaded) {
                        modelStatus.className = 'model-status model-ready';
                        modelStatus.innerHTML = '✅ Model พร้อมใช้งาน - คลาส: ' + Object.values(data.classes).join(', ');
                    } else {
                        modelStatus.className = 'model-status model-error';
                        modelStatus.innerHTML = '❌ Model ยังไม่พร้อม - กรุณาตรวจสอบ logs';
                    }
                })
                .catch(error => {
                    modelStatus.className = 'model-status model-error';
                    modelStatus.innerHTML = '❌ ไม่สามารถเชื่อมต่อ Backend ได้';
                });
        }

        // File input change event
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                uploadArea.innerHTML = `<p>✅ เลือกไฟล์: ${selectedFile.name}</p>`;
                processBtn.disabled = false;
            }
        });

        // Drag and drop events
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                selectedFile = e.dataTransfer.files[0];
                fileInput.files = e.dataTransfer.files;
                uploadArea.innerHTML = `<p>✅ เลือกไฟล์: ${selectedFile.name}</p>`;
                processBtn.disabled = false;
            }
        });

        async function processImage() {
            if (!selectedFile) return;

            // Show loading
            loading.style.display = 'block';
            results.style.display = 'none';
            processBtn.disabled = true;

            try {
                // Convert file to base64
                const base64 = await fileToBase64(selectedFile);
                
                // Send to server
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: base64
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    displayResults(result, base64);
                } else {
                    showError('เกิดข้อผิดพลาด: ' + result.error);
                }
            } catch (error) {
                showError('เกิดข้อผิดพลาดในการเชื่อมต่อ: ' + error.message);
            } finally {
                loading.style.display = 'none';
                processBtn.disabled = false;
            }
        }

        function displayResults(result, originalImageB64) {
            // Show images
            document.getElementById('originalImage').src = originalImageB64;
            document.getElementById('annotatedImage').src = result.annotated_image;
            
            // Show detection info
            let infoHtml = `<h4>🎯 พบวัตถุ ${result.total_detections} ชิ้น</h4>`;
            
            result.detections.forEach((detection, index) => {
                infoHtml += `
                    <div class="detection-item">
                        <strong>${detection.class}</strong> 
                        (ความแม่นยำ: ${(detection.confidence * 100).toFixed(1)}%)
                        <br>
                        <small>ตำแหน่ง: [${detection.bbox.join(', ')}]</small>
                    </div>
                `;
            });
            
            document.getElementById('detectionInfo').innerHTML = infoHtml;
            results.style.display = 'block';
        }

        function showError(message) {
            results.innerHTML = `<div class="error">${message}</div>`;
            results.style.display = 'block';
        }

        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result);
                reader.onerror = error => reject(error);
            });
        }

        // Refresh model status every 30 seconds
        setInterval(checkModelStatus, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    # ใช้ไฟล์ HTML แยกต่างหาก หรือใช้ template string
    try:
        # ถ้ามีไฟล์ templates/index.html
        return render_template('index.html')
    except:
        # ถ้าไม่มี ใช้ template string
        return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded yet. Please wait or check logs.'})
        
        result = process_image(data['image'])
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': class_names if class_names else {},
        'model_path': 'models/best.pt' if os.path.exists('models/best.pt') else None
    })

@app.route('/download-model', methods=['POST'])
def force_download_model():
    """Force re-download model"""
    try:
        global model, class_names
        
        # Remove existing model
        if os.path.exists('models/best.pt'):
            os.remove('models/best.pt')
        
        # Clear global variables
        model = None
        class_names = []
        
        # Try to download again
        if load_model():
            return jsonify({'success': True, 'message': 'Model downloaded and loaded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to download model'})
            
    except Exception as e:
        logger.error(f"Error in force download: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Flask app...")
    if load_model():
        logger.info("Model loaded successfully. Starting server...")
    else:
        logger.warning("Model loading failed. Server will start but predictions won't work until model is loaded.")
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
