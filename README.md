# 🔍 YOLO Object Detection Web App

A modern web application for object detection using custom-trained YOLO models, built with Flask and deployable on Railway.

## ✨ Features

- 🎯 **Custom YOLO Model Support** - Use your own trained models
- 🌐 **Web Interface** - Beautiful, responsive UI with drag & drop
- 📱 **Mobile Friendly** - Works on all devices
- ⚡ **Auto Model Download** - Automatically downloads models from Google Drive
- 🚀 **One-Click Deploy** - Easy deployment on Railway
- 📊 **Real-time Detection** - Instant object detection with confidence scores
- 🎨 **Visual Results** - Side-by-side comparison with bounding boxes

## 🛠️ Tech Stack

- **Backend**: Flask, OpenCV, Ultralytics YOLO
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Railway, Docker
- **Model Storage**: Google Drive auto-download

## 🚀 Quick Start

### 1. Prepare Your Model

1. Upload your trained YOLO model (`best.pt`) to Google Drive
2. Set sharing to "Anyone with the link can view"
3. Copy the File ID from the URL:
   ```
   https://drive.google.com/file/d/FILE_ID_HERE/view
   ```

### 2. Deploy on Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template)

1. Click the deploy button above
2. Connect your GitHub repository
3. Set environment variables:
   ```
   GDRIVE_FILE_ID=your_google_drive_file_id
   CONFIDENCE_THRESHOLD=0.5
   ```
4. Deploy and wait for the model to download

### 3. Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/yolo-web-app.git
cd yolo-web-app

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GDRIVE_FILE_ID=your_file_id
export CONFIDENCE_THRESHOLD=0.5

# Run the application
python app.py
```

## 📋 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GDRIVE_FILE_ID` | Google Drive File ID of your YOLO model | Required |
| `MODEL_URL` | Direct URL to model file (alternative to Google Drive) | Optional |
| `CONFIDENCE_THRESHOLD` | Detection confidence threshold (0.0-1.0) | `0.5` |
| `PORT` | Server port | `8080` |

## 🔧 API Endpoints

### `GET /`
Main web interface

### `POST /predict`
Object detection API
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 50, 300, 400]
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "total_detections": 1
}
```

### `GET /health`
Health check and model status
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": {"0": "person", "1": "car"},
  "model_path": "models/best.pt"
}
```

### `POST /download-model`
Force model re-download

## 📁 Project Structure

```
yolo-web-app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── railway.json          # Railway deployment config
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── templates/           # Frontend templates (optional)
    └── index.html       # Alternative HTML template
```

## 🎯 Model Requirements

Your YOLO model should be:
- Trained with Ultralytics YOLO (YOLOv8)
- Saved as `.pt` format
- Compatible with `ultralytics` Python package

Supported formats:
- `best.pt` (recommended)
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`

## 🔄 Model Management

### Update Your Model
1. Upload new model to Google Drive
2. Update `GDRIVE_FILE_ID` in Railway
3. Restart the application or call `/download-model`

### Use Different Model Sources
```bash
# Google Drive (recommended)
GDRIVE_FILE_ID=1ABCDEFghijklmnopQRSTUV123456789

# Direct URL
MODEL_URL=https://github.com/user/repo/releases/download/v1.0/best.pt

# Hugging Face Hub
MODEL_HF_REPO=username/model-name
```

## 📱 Usage

1. **Open the web app** in your browser
2. **Upload an image** by clicking or dragging
3. **Click "วิเคราะห์รูปภาพ"** to run detection
4. **View results** with bounding boxes and confidence scores

## ⚡ Performance Tips

- Use smaller models (YOLOv8n) for faster inference
- Resize large images before upload
- Adjust confidence threshold based on your needs
- Monitor Railway resource usage

## 🐛 Troubleshooting

### Model Not Loading
```bash
# Check logs
railway logs

# Verify File ID
railway variables

# Force re-download
curl -X POST https://your-app.railway.app/download-model
```

### Common Issues
1. **Google Drive file not public** - Make sure sharing is enabled
2. **File ID incorrect** - Double-check the ID from Google Drive URL
3. **Model incompatible** - Ensure it's trained with Ultralytics YOLO
4. **Memory limit exceeded** - Use smaller model or upgrade Railway plan

## 📊 Monitoring

Check your app status:
- **Health endpoint**: `/health`
- **Railway logs**: `railway logs`
- **Model status**: Displayed on the main page

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection model
- [Railway](https://railway.app) - Deployment platform
- [Flask](https://flask.palletsprojects.com/) - Web framework

## 📞 Support

If you encounter any issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review Railway deployment logs
3. Open an issue on GitHub

---

**Made with ❤️ for the YOLO community**
