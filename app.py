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
    """แยก file ID จาก Google Drive URL หลายรูปแบบ"""
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
    """ดาวน์โหลด model จาก Google Drive with multiple methods - ปรับปรุงแล้ว"""
    model_path = 'models/best.pt'
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return True
        
    try:
        os.makedirs('models', exist_ok=True)
        logger.info("Downloading model from Google Drive...")
        
        # รับ file ID จาก environment variable
        gdrive_input = os.getenv('GDRIVE_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID')
        
        if gdrive_input == 'YOUR_GOOGLE_DRIVE_FILE_ID':
            logger.error("Please set GDRIVE_FILE_ID environment variable")
            return False
        
        # ถ้าเป็น URL เต็ม ให้แยก file ID ออกมา
        if gdrive_input.startswith('http'):
            file_id = extract_file_id_from_url(gdrive_input)
            if not file_id:
                logger.error(f"Cannot extract file ID from URL: {gdrive_input}")
                return False
            logger.info(f"Extracted file ID: {file_id}")
        else:
            file_id = gdrive_input
            
        # ลองหลายวิธี
        download_methods = [
            # Method 1: gdown with fuzzy matching
            lambda: gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False, fuzzy=True),
            # Method 2: gdown with direct download
            lambda: gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", model_path, quiet=False),
            # Method 3: requests with session handling
            lambda: download_with_requests(file_id, model_path),
        ]
        
        for i, method in enumerate(download_methods, 1):
            try:
                logger.info(f"Trying download method {i}...")
                method()
                
                # ตรวจสอบว่าไฟล์ถูกดาวน์โหลดจริง
                if os.path.exists(model_path) and os.path.getsize(model_path) > 100000:  # อย่างน้อย 100KB
                    logger.info(f"Method {i} successful! File size: {os.path.getsize(model_path)} bytes")
                    
                    # ตรวจสอบว่าไฟล์เป็น PyTorch model จริงหรือไม่
                    if is_valid_pytorch_model(model_path):
                        logger.info("Valid PyTorch model detected!")
                        return True
                    else:
                        logger.warning(f"Downloaded file is not a valid PyTorch model")
                        os.remove(model_path)
                else:
                    logger.warning(f"Method {i} failed - file not created or too small")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        
            except Exception as e:
                logger.warning(f"Method {i} failed: {str(e)}")
                if os.path.exists(model_path):
                    os.remove(model_path)
                continue
        
        logger.error("All download methods failed")
        return False
        
    except Exception as e:
        logger.error(f"Error downloading model from Google Drive: {str(e)}")
        return False

def is_valid_pytorch_model(file_path):
    """ตรวจสอบว่าไฟล์เป็น PyTorch model จริงหรือไม่"""
    try:
        # อ่านไบต์แรกของไฟล์
        with open(file_path, 'rb') as f:
            header = f.read(10)
        
        # PyTorch model จะขึ้นต้นด้วย PK (ZIP format) หรือ magic bytes อื่นๆ
        # HTML file จะขึ้นต้นด้วย < หรือ <!
        if header.startswith(b'<') or header.startswith(b'<!'):
            logger.error("File appears to be HTML, not a PyTorch model")
            return False
            
        # ตรวจสอบขนาดไฟล์ (PyTorch model ปกติจะมีขนาดใหญ่กว่า 1MB)
        file_size = os.path.getsize(file_path)
        if file_size < 1000000:  # น้อยกว่า 1MB
            logger.warning(f"File size ({file_size} bytes) seems too small for a YOLO model")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating model file: {str(e)}")
        return False

def download_with_requests(file_id, output_path):
    """ดาวน์โหลดด้วย requests library - ปรับปรุงแล้ว"""
    import requests
    
    # URL สำหรับดาวน์โหลดจาก Google Drive
    urls = [
        f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1",
        f"https://drive.google.com/uc?id={file_id}&export=download&confirm=1",
        f"https://drive.google.com/uc?export=download&id={file_id}",
    ]
    
    session = requests.Session()
    
    for url in urls:
        try:
            logger.info(f"Trying requests with URL: {url}")
            
            # ดาวน์โหลดครั้งแรก
            response = session.get(url, stream=True, allow_redirects=True)
            
            # ตรวจสอบว่าเป็น virus scan warning page หรือไม่
            if response.headers.get('content-type', '').startswith('text/html'):
                content_preview = response.content[:1000].decode('utf-8', errors='ignore')
                
                # ถ้าเจอ warning page ให้หา download link
                if 'virus scan warning' in content_preview.lower() or 'download anyway' in content_preview.lower():
                    logger.info("Detected virus scan warning page, looking for download link...")
                    
                    # หา download link ในหน้า HTML
                    download_link_patterns = [
                        r'href="([^"]*&amp;confirm=[^"]*)"',
                        r'href="([^"]*confirm=[^"]*)"',
                        r"href='([^']*confirm=[^']*)'",
                    ]
                    
                    for pattern in download_link_patterns:
                        matches = re.findall(pattern, content_preview)
                        for match in matches:
                            if 'export=download' in match:
                                # แปลง &amp; เป็น &
                                clean_url = match.replace('&amp;', '&')
                                if not clean_url.startswith('http'):
                                    clean_url = 'https://drive.google.com' + clean_url
                                
                                logger.info(f"Found download link: {clean_url}")
                                response = session.get(clean_url, stream=True, allow_redirects=True)
                                break
                    else:
                        continue  # ไม่เจอ link ให้ลอง URL ถัดไป
            
            # ตรวจสอบ response
            if response.status_code == 200:
                # ตรวจสอบ content type
                content_type = response.headers.get('content-type', '')
                if content_type.startswith('text/html'):
                    logger.warning(f"Received HTML content instead of binary file")
                    continue
                
                # ดาวน์โหลดไฟล์
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
                
                print()  # New line
                
                # ตรวจสอบขนาดไฟล์
                if os.path.getsize(output_path) > 100000:  # มากกว่า 100KB
                    logger.info(f"Successfully downloaded {downloaded} bytes")
                    return True
                else:
                    logger.warning("Downloaded file too small")
                    os.remove(output_path)
                    
        except Exception as e:
            logger.warning(f"Requests method failed with URL {url}: {str(e)}")
            continue
            
    return False

def download_model_from_url():
    """ดาวน์โหลด model จาก URL (ปรับปรุงแล้ว)"""
    model_path = 'models/best.pt'
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return True
        
    try:
        os.makedirs('models', exist_ok=True)
        
        # รับ URL จาก environment variable
        model_url = os.getenv('MODEL_URL')
        
        if not model_url:
            logger.info("MODEL_URL environment variable not set, skipping URL download")
            return False
            
        logger.info(f"Downloading model from URL: {model_url}")
        
        # ถ้า URL เป็น Google Drive link ให้แปลงเป็น direct download
        if 'drive.google.com' in model_url:
            file_id = extract_file_id_from_url(model_url)
            if file_id:
                model_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"
                logger.info(f"Converted to direct download URL: {model_url}")
        
        response = requests.get(model_url, stream=True, allow_redirects=True)
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
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line
        
        # ตรวจสอบว่าไฟล์ดาวน์โหลดสำเร็จ
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
    """Load YOLO model with enhanced error handling"""
    global model, class_names
    
    try:
        model_path = 'models/best.pt'
        
        # แสดงสถานะ environment variables
        gdrive_id = os.getenv('GDRIVE_FILE_ID', 'NOT_SET')
        model_url = os.getenv('MODEL_URL', 'NOT_SET')
        logger.info(f"GDRIVE_FILE_ID: {'SET' if gdrive_id != 'NOT_SET' else 'NOT_SET'}")
        logger.info(f"MODEL_URL: {'SET' if model_url != 'NOT_SET' else 'NOT_SET'}")
        
        # ลองดาวน์โหลดจาก Google Drive ก่อน
        if not os.path.exists(model_path):
            logger.info("Model file not found, attempting download...")
            if not download_model_from_gdrive():
                logger.info("Google Drive download failed, trying URL method...")
                if not download_model_from_url():
                    logger.error("Failed to download model from all sources")
                    return False
        
        # ตรวจสอบไฟล์ก่อน load
        if not os.path.exists(model_path):
            logger.error(f"Model file still not found at {model_path}")
            return False
            
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file found: {model_path} (size: {file_size} bytes)")
        
        if not is_valid_pytorch_model(model_path):
            logger.error("Model file validation failed")
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
