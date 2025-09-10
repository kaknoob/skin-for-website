# app.py
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# โหลด YOLO model
model = YOLO('yolov8n.pt')  # ใช้ model ที่เล็กสุดเพื่อความเร็ว

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูลรูปภาพจาก request
        data = request.json
        image_data = data['image'].split(',')[1]  # ลบ data:image/jpeg;base64,
        
        # แปลง base64 เป็น image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # ทำ prediction
        results = model(image_cv)
        
        # แปลงผลลัพธ์
        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # ข้อมูล bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # วาด bounding boxes บนรูปภาพ
        annotated_image = results[0].plot()
        
        # แปลงรูปภาพที่ annotate แล้วเป็น base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'annotated_image': f"data:image/jpeg;base64,{img_base64}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
