# User Manual and Deployment Guide
## Real-time Facial Expression Recognition System

### Version 1.0 | August 2025

---

## Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [System Requirements](#system-requirements)
3. [Installation Instructions](#installation-instructions)
4. [User Interface Guide](#user-interface-guide)
5. [Configuration Options](#configuration-options)
6. [Deployment Scenarios](#deployment-scenarios)
7. [Maintenance and Updates](#maintenance-and-updates)
8. [Frequently Asked Questions](#frequently-asked-questions)

---

## 1. Quick Start Guide

### 1.1 For Non-Technical Users

#### Step 1: Download and Extract
1. Download the project folder to your Desktop
2. Extract all files if compressed
3. Open Command Prompt or PowerShell

#### Step 2: Navigate to Project
```cmd
cd "C:\Users\[YourUsername]\OneDrive\Desktop\Real-time-Facial-Expression-Recognition-System"
```

#### Step 3: Run the Application
```cmd
python src/face_detect.py
```

#### Step 4: Use the System
1. **Camera Permission**: Allow camera access if prompted
2. **Position Yourself**: Sit 2-3 feet from camera with good lighting
3. **View Results**: See emotion labels and confidence scores on screen
4. **Exit**: Press 'q' key to quit

### 1.2 Expected Behavior
- **Startup Time**: 5-10 seconds for model loading
- **Frame Rate**: 20-30 FPS on modern computers
- **Detection**: Blue boxes around faces with green emotion labels
- **Accuracy**: 85%+ in good lighting conditions

---

## 2. System Requirements

### 2.1 Minimum Requirements
| Component | Specification |
|-----------|---------------|
| **OS** | Windows 10, macOS 10.14, Ubuntu 18.04 |
| **Python** | Version 3.8+ |
| **RAM** | 8 GB |
| **Storage** | 2 GB free space |
| **Camera** | 720p webcam or built-in camera |
| **CPU** | Intel i5 / AMD Ryzen 5 equivalent |

### 2.2 Recommended Requirements
| Component | Specification |
|-----------|---------------|
| **OS** | Windows 11, macOS 12+, Ubuntu 20.04+ |
| **Python** | Version 3.11 |
| **RAM** | 16 GB |
| **Storage** | 5 GB free space |
| **Camera** | 1080p webcam |
| **CPU** | Intel i7 / AMD Ryzen 7 |
| **GPU** | NVIDIA GTX 1060 / AMD RX 580 (optional) |

### 2.3 Network Requirements
- **Internet Connection**: Required for initial setup and model downloads
- **Bandwidth**: 10 Mbps for downloading dependencies (~500 MB)
- **Offline Operation**: Supported after initial setup

---

## 3. Installation Instructions

### 3.1 Automated Installation (Recommended)

#### For Windows Users:
1. **Download Python**: Visit [python.org](https://python.org) and install Python 3.11
2. **Open PowerShell as Administrator**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. **Run Installation Script**:
   ```powershell
   cd "Real-time-Facial-Expression-Recognition-System"
   .\install.bat
   ```

#### For macOS Users:
1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. **Install Python**:
   ```bash
   brew install python@3.11
   ```
3. **Run Installation**:
   ```bash
   cd Real-time-Facial-Expression-Recognition-System
   chmod +x install.sh
   ./install.sh
   ```

### 3.2 Manual Installation

#### Step 1: Create Virtual Environment
```bash
python -m venv fer_env
```

#### Step 2: Activate Environment
**Windows:**
```cmd
fer_env\Scripts\activate
```
**macOS/Linux:**
```bash
source fer_env/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install opencv-python==4.12.0.88
pip install fer==22.5.1
pip install tensorflow>=2.0
pip install moviepy==1.0.3
pip install numpy>=1.25.0
```

#### Step 4: Verify Installation
```bash
python quick_test.py
```

### 3.3 Troubleshooting Installation

#### Common Issues and Solutions:

**Issue 1: Python not found**
```
Error: 'python' is not recognized as an internal or external command
```
**Solution**: Add Python to system PATH or use full path to python.exe

**Issue 2: Permission denied**
```
Error: [Errno 13] Permission denied
```
**Solution**: Run command prompt as administrator

**Issue 3: Network timeout**
```
Error: Read timed out
```
**Solution**: Use alternative package index:
```bash
pip install -i https://pypi.org/simple/ tensorflow
```

---

## 4. User Interface Guide

### 4.1 Main Window Layout

```
┌─────────────────────────────────────────────────────┐
│ Real-time Expression Recognition                    │
├─────────────────────────────────────────────────────┤
│  detect 2 face                            [Counter] │
│                                                     │
│     ┌─────────────┐        ┌─────────────┐          │
│     │ happy (0.87)│        │ sad (0.92)  │          │
│     │             │        │             │          │
│     │   [Face 1]  │        │   [Face 2]  │          │
│     │             │        │             │          │
│     └─────────────┘        └─────────────┘          │
│                                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
Press 'q' to quit
```

### 4.2 Visual Elements Explained

#### 4.2.1 Face Detection Indicators
- **Blue Rectangle**: Indicates detected face boundary
- **Rectangle Size**: Adjusts to face size automatically
- **Multiple Faces**: System can detect up to 10 faces simultaneously

#### 4.2.2 Emotion Labels
- **Text Format**: `emotion_name (confidence_score)`
- **Color**: Green text for visibility
- **Position**: Above each detected face
- **Update Rate**: 30 times per second

#### 4.2.3 Statistics Display
- **Face Counter**: Shows total faces in current frame
- **Position**: Top-left corner of window
- **Format**: "detect X face" or "detect X faces"

### 4.3 Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| **q** | Quit | Exit application |
| **Space** | Pause/Resume | Pause video processing |
| **s** | Screenshot | Save current frame |
| **r** | Reset | Reset detection parameters |
| **h** | Help | Show help overlay |

### 4.4 Mouse Interactions

| Action | Function |
|--------|----------|
| **Left Click** | Focus on clicked face region |
| **Right Click** | Show emotion details popup |
| **Scroll Wheel** | Zoom in/out (if implemented) |
| **Double Click** | Toggle fullscreen mode |

---

## 5. Configuration Options

### 5.1 Basic Configuration

#### 5.1.1 Camera Settings
Create a file called `config.json`:
```json
{
    "camera": {
        "device_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "detection": {
        "confidence_threshold": 0.5,
        "min_face_size": 30,
        "max_faces": 10
    },
    "display": {
        "window_title": "Emotion Recognition",
        "show_confidence": true,
        "show_fps": true
    }
}
```

#### 5.1.2 Performance Tuning
```json
{
    "performance": {
        "frame_skip": 1,
        "use_gpu": true,
        "batch_processing": false,
        "model_precision": "float32"
    }
}
```

### 5.2 Advanced Configuration

#### 5.2.1 Model Selection
```json
{
    "models": {
        "face_detector": "mtcnn",
        "emotion_model": "fer2013",
        "model_path": "./models/",
        "download_if_missing": true
    }
}
```

#### 5.2.2 Output Options
```json
{
    "output": {
        "save_frames": false,
        "output_directory": "./outputs/",
        "log_detections": true,
        "export_statistics": true
    }
}
```

### 5.3 Environment Variables

Set these before running the application:

#### Windows:
```cmd
set FER_MODEL_PATH=C:\path\to\models
set FER_CONFIDENCE_THRESHOLD=0.7
set FER_DEBUG_MODE=true
```

#### macOS/Linux:
```bash
export FER_MODEL_PATH=/path/to/models
export FER_CONFIDENCE_THRESHOLD=0.7
export FER_DEBUG_MODE=true
```

---

## 6. Deployment Scenarios

### 6.1 Single Computer Deployment

#### 6.1.1 Desktop Application
```bash
# Create standalone executable
pip install pyinstaller
pyinstaller --onefile --windowed src/face_detect.py
```

#### 6.1.2 Kiosk Mode
```python
# Add to face_detect.py for kiosk deployment
cv2.namedWindow('FER System', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('FER System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
```

### 6.2 Multi-Camera Deployment

#### 6.2.1 Multiple Camera Script
```python
# multi_camera_fer.py
import threading
import cv2
from fer import FER

class MultiCameraFER:
    def __init__(self, camera_ids=[0, 1, 2]):
        self.camera_ids = camera_ids
        self.fer_detectors = [FER(mtcnn=True) for _ in camera_ids]
        self.cameras = []
        
    def start_cameras(self):
        for i, camera_id in enumerate(self.camera_ids):
            thread = threading.Thread(
                target=self.process_camera, 
                args=(camera_id, self.fer_detectors[i])
            )
            thread.daemon = True
            thread.start()
            
    def process_camera(self, camera_id, fer_detector):
        cap = cv2.VideoCapture(camera_id)
        
        while True:
            ret, frame = cap.read()
            if ret:
                results = fer_detector.detect_emotions(frame)
                # Process results for camera_id
                self.display_results(camera_id, frame, results)
```

### 6.3 Network Deployment

#### 6.3.1 Web Service API
```python
# fer_api.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from fer import FER

app = Flask(__name__)
fer_detector = FER(mtcnn=True)

@app.route('/detect_emotions', methods=['POST'])
def detect_emotions_api():
    # Receive image data
    file = request.files['image']
    
    # Convert to OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Detect emotions
    results = fer_detector.detect_emotions(img)
    
    return jsonify({
        'status': 'success',
        'detections': results,
        'face_count': len(results)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### 6.3.2 Client Application
```python
# fer_client.py
import requests
import cv2

def send_frame_to_server(frame, server_url='http://localhost:5000'):
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    
    # Send to server
    files = {'image': buffer.tobytes()}
    response = requests.post(f'{server_url}/detect_emotions', files=files)
    
    return response.json()
```

### 6.4 Cloud Deployment

#### 6.4.1 Docker Container
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port for web service
EXPOSE 5000

# Start application
CMD ["python", "src/fer_api.py"]
```

#### 6.4.2 Kubernetes Deployment
```yaml
# fer-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fer-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fer-system
  template:
    metadata:
      labels:
        app: fer-system
    spec:
      containers:
      - name: fer-container
        image: fer-system:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

---

## 7. Maintenance and Updates

### 7.1 Regular Maintenance Tasks

#### 7.1.1 Weekly Maintenance
- **Check System Performance**: Monitor FPS and accuracy
- **Clean Temporary Files**: Remove cached model files if disk space low
- **Update Dependencies**: Check for security updates
- **Backup Configuration**: Save custom configuration files

#### 7.1.2 Monthly Maintenance
- **Model Updates**: Check for newer pre-trained models
- **Performance Analysis**: Review detection accuracy logs
- **Hardware Health**: Check camera and system status
- **Documentation Updates**: Update user manuals if needed

### 7.2 Update Procedures

#### 7.2.1 Software Updates
```bash
# Update Python packages
pip list --outdated
pip install --upgrade opencv-python fer tensorflow

# Update system
python update_system.py
```

#### 7.2.2 Model Updates
```bash
# Download latest models
python download_models.py --model-version latest

# Backup current models
cp -r models/ models_backup/

# Install new models
python install_models.py --model-path ./new_models/
```

### 7.3 Backup and Recovery

#### 7.3.1 Backup Script
```python
# backup_system.py
import shutil
import datetime
import os

def backup_system():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup configuration
    shutil.copy2("config.json", backup_dir)
    
    # Backup custom models
    if os.path.exists("models/custom/"):
        shutil.copytree("models/custom/", f"{backup_dir}/models")
    
    # Backup logs
    if os.path.exists("logs/"):
        shutil.copytree("logs/", f"{backup_dir}/logs")
    
    print(f"Backup completed: {backup_dir}")

if __name__ == "__main__":
    backup_system()
```

### 7.4 Performance Monitoring

#### 7.4.1 System Metrics
```python
# monitor_performance.py
import psutil
import time
import logging

def monitor_system():
    # Setup logging
    logging.basicConfig(
        filename='performance.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    while True:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU usage (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0
        except:
            gpu_usage = 0
        
        # Log metrics
        logging.info(f"CPU: {cpu_percent}% | RAM: {memory_percent}% | GPU: {gpu_usage}%")
        
        time.sleep(60)  # Log every minute
```

---

## 8. Frequently Asked Questions

### 8.1 Installation Issues

**Q: The system says "camera not found" but I have a webcam**
**A:** Try these solutions:
1. Close other applications using the camera (Skype, Teams, etc.)
2. Check camera permissions in system settings
3. Try changing camera index in code from 0 to 1
4. Restart your computer and try again

**Q: Getting "ModuleNotFoundError" when running the code**
**A:** This means Python packages aren't installed correctly:
1. Activate your virtual environment: `venv\Scripts\activate`
2. Reinstall packages: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.8+)

**Q: Installation is very slow or times out**
**A:** Try these approaches:
1. Use a different package index: `pip install -i https://pypi.org/simple/ package_name`
2. Install packages one by one instead of using requirements.txt
3. Check your internet connection
4. Try installing during off-peak hours

### 8.2 Performance Issues

**Q: The system is running very slowly (low FPS)**
**A:** Optimize performance with these steps:
1. Close unnecessary applications
2. Ensure good lighting (reduces processing time)
3. Move closer to camera (makes face detection easier)
4. Enable GPU acceleration if available
5. Reduce video resolution in camera settings

**Q: Emotion detection is inaccurate**
**A:** Improve accuracy by:
1. Ensuring good, even lighting on your face
2. Looking directly at the camera
3. Keeping a neutral background
4. Avoiding extreme facial angles
5. Making sure face is clearly visible (no obstructions)

**Q: Multiple faces are not detected properly**
**A:** For multi-face scenarios:
1. Ensure all faces are well-lit
2. Keep faces at similar distances from camera
3. Avoid overlapping faces
4. Use higher resolution camera if available

### 8.3 Technical Questions

**Q: Can I use this system for commercial purposes?**
**A:** Check the licenses of included libraries:
- OpenCV: Apache 2.0 License (commercial use allowed)
- FER library: Check specific license terms
- TensorFlow: Apache 2.0 License (commercial use allowed)
- Always verify current license terms before commercial deployment

**Q: How accurate is the emotion detection?**
**A:** Accuracy depends on several factors:
- Controlled environment: 85-90% accuracy
- Real-world conditions: 70-80% accuracy
- Factors affecting accuracy: lighting, face angle, image quality, facial expressions intensity

**Q: Can I train the system with my own data?**
**A:** Yes, you can:
1. Collect and label your own emotion dataset
2. Use transfer learning with existing models
3. Fine-tune models for specific demographics or use cases
4. Refer to the technical documentation for training procedures

**Q: What emotions can the system detect?**
**A:** The system detects 7 basic emotions:
- Happy (joy, contentment)
- Sad (sorrow, melancholy)
- Angry (frustration, rage)
- Fear (anxiety, worry)
- Surprise (astonishment, shock)
- Disgust (distaste, revulsion)
- Neutral (no specific emotion)

### 8.4 Customization Questions

**Q: Can I change the colors of the bounding boxes and text?**
**A:** Yes, modify these lines in `face_detect.py`:
```python
# Change bounding box color (BGR format)
cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue
# To red: (0, 0, 255)
# To green: (0, 255, 0)

# Change text color
cv2.putText(frame, text, position, font, scale, (0, 255, 0), thickness)  # Green
```

**Q: Can I save the detected emotions to a file?**
**A:** Yes, add logging functionality:
```python
import csv
import datetime

# Add this in your main loop
with open('emotions_log.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    timestamp = datetime.datetime.now()
    writer.writerow([timestamp, emotion, score])
```

**Q: How do I change the confidence threshold?**
**A:** Modify the detection threshold in the FER initialization:
```python
# In face_detect.py, you can filter results by confidence
if score > 0.7:  # Only show emotions with >70% confidence
    # Display emotion
```

---

## Conclusion

This user manual provides comprehensive guidance for installing, configuring, and using the Real-time Facial Expression Recognition System. The system is designed to be user-friendly while offering extensive customization options for advanced users.

### Quick Reference:
- **Start Application**: `python src/face_detect.py`
- **Quit Application**: Press 'q' key
- **Best Performance**: Good lighting, 2-3 feet from camera
- **Support**: Refer to troubleshooting section or technical documentation

### Getting Help:
- Check the FAQ section first
- Review error messages for specific guidance
- Consult technical documentation for advanced issues
- Ensure all system requirements are met

This system represents a practical implementation of state-of-the-art emotion recognition technology, suitable for research, education, and commercial applications.

---

*User Manual Version: 1.0*  
*Last Updated: August 1, 2025*  
*Support Contact: Technical Team*
