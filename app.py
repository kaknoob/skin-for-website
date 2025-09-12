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
    """ดาวน์โหลด model จาก Google Drive (เสถียรขึ้น)"""
    import gdown

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

        # แยก file_id จาก URL
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

