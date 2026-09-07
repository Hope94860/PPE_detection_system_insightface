# SiteSecureVision: Face & PPE Detection System

SiteSecureVision is an advanced real-time monitoring system designed to enhance safety and security in industrial and construction environments. Leveraging state-of-the-art AI models, the system provides simultaneous face recognition, Personal Protective Equipment (PPE) compliance monitoring, and incident detection.

## 🚀 Features

- **Real-time Face Recognition**: High-performance recognition using InsightFace, supporting multi-camera streams.
- **PPE Compliance Monitoring**: Automatically detects safety helmets, reflective vests, gloves, boots, and goggles.
- **Emergency Detection**: 
  - Fire and smoke detection using specialized models.
  - Fall detection via pose estimation (YOLOv8-pose).
  - Night-time intruder detection.
- **ALPR (Automatic License Plate Recognition)**: Automated vehicle entry/exit logging with Indian plate format support.
- **Live Monitoring Dashboard**: Multi-camera view with real-time compliance overlays and person counts.
- **Incident Recording**: Automatic recording and logging of safety violations and emergency events.
- **Notifications**: Automated alerts via WhatsApp/SMS for critical incidents.

## 🛠️ Technology Stack

- **Core**: Python 3.x
- **Inference**: ONNX Runtime (CPU/GPU acceleration)
- **Object Detection**: YOLOv8
- **Face Recognition**: InsightFace (Buffalo_L)
- **Database**: MongoDB
- **UI**: Custom Desktop Dashboard (PyQt/PySide)
- **OCR**: EasyOCR / YOLO OCR

## 📋 Prerequisites

- **Python**: 3.10+
- **Database**: MongoDB (Local or Atlas)
- **Hardware**: NVIDIA GPU with CUDA support recommended for real-time performance on multiple streams.

## ⚙️ Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Amoomo321/Face_PPE_detection_system_insightface.git
   cd Face_PPE_detection_system_insightface
   ```

2. **Initialize Environment**:
   Run the included PowerShell script to set up prerequisites and the virtual environment:
   ```powershell
   ./setup_prerequisites.ps1
   ```

3. **Configure Database**:
   Ensure MongoDB is running and update the connection string in `mongo_db_manager.py` if necessary.

4. **Launch Application**:
   ```bash
   python launcher.py
   ```

## 🔒 Security Note

Sensitive files such as RSA private keys (`*.pem`), license files (`*.lic`), and large model weights (`*.onnx`) are excluded from this repository for security and performance. Please contact the developer for access to these assets.

## 📄 License

This project is licensed under the terms of the private commercial license provided within the system.

---
Developed by [Amoomo321](https://github.com/Amoomo321)
