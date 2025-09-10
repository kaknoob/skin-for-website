from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
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

def load_model():
    """Load YOLO model"""
    global model, class_names
    try:
        # ใส่ path ของ model ที่เทรนแล้ว
        model_path = os.getenv('MODEL_PATH', 'best.pt')  # ใช้ environment variable
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
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
                    if confidence > 0.5:  # สามารถปรับได้
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

# HTML Template
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 YOLO Object Detection</h1>
        
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
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
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
        'classes': class_names if class_names else []
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask app...")
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Failed to load model. Exiting...")
