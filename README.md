# Face Recognition System

> A lightweight, dependency-free face recognition system powered by OpenCV's DNN module.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-3776ab?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

</div>

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

✨ **Real-time Detection** — Identify faces in live video streams  
📸 **Image Processing** — Analyze faces in static images  
⚡ **Zero Setup** — Models download automatically on first run  
💪 **CPU-Only** — No GPU required  
🔌 **Pure OpenCV** — No TensorFlow, no compilation, no complications  
👥 **Multi-Face** — Detect and recognize multiple people simultaneously  

## Requirements

| Package | Version |
|---------|---------|
| Python | 3.7+ |
| OpenCV | 4.9.0 |
| NumPy | 1.26.3 |

## Quick Start

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
pip install -r requirements.txt
```

### 2️⃣ Prepare Training Data

Create a `faces/` directory with images of people you want to recognize:

```
faces/
├── john.jpg
├── sarah.png
└── alex.jpg
```

> **Tip:** Use filename as the person's name (e.g., `john.jpg` → recognized as "John")

### 3️⃣ Run the Application

```bash
python main.py
```

Select your mode:
- **Mode 1:** Real-time webcam recognition (press `q` to quit)
- **Mode 2:** Process a single image file

## Usage

## Usage

<details open>
<summary><b>Real-time Webcam Recognition</b></summary>

```bash
python main.py
# Select option 1
# Press 'q' to quit
```

</details>

<details>
<summary><b>Image File Processing</b></summary>

```bash
python main.py
# Select option 2
# Provide your image path
```

</details>

## How It Works

```
Model Setup (auto-download on first run)
    ↓
Directory Scan (faces/ folder)
    ↓
Face Detection (OpenCV DNN)
    ↓
Face Recognition (LBPH algorithm)
    ↓
Results Display (with confidence scores)
```

### Technical Details

| Component | Technology |
|-----------|-----------|
| Face Detection | OpenCV DNN + SSD (Single Shot Detector) |
| Face Recognition | LBPH (Local Binary Patterns Histograms) |
| Pre-trained Models | Caffe format (~10MB, auto-downloaded) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Detection Speed | ~30 FPS (modern CPU) |
| Recognition Accuracy | Good for controlled environments |
| Memory Usage | ~100MB (including models) |

## Configuration

Customize the system by modifying `face_recognition.py`:

| Parameter | Location | Default | Purpose |
|-----------|----------|---------|---------|
| Detection Confidence | Line 52 | `0.5` | Face detection threshold |
| Recognition Threshold | Line 137 | `100` | Face match confidence |
| Face ROI Size | Line 95 | `(200, 200)` | Cropped face dimensions |

## Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Submit** a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- 🖤 [OpenCV](https://opencv.org/) — For the incredible computer vision library
- 📦 OpenCV model zoo — For pre-trained face detection models

## Disclaimer

⚠️ This is an educational face recognition system designed for simple use cases. For production applications requiring high accuracy, consider using advanced models like **FaceNet**, **ArcFace**, or commercial solutions.

---

<div align="center">

**[↑ back to top](#face-recognition-system)**

Made with ❤️ using [OpenCV](https://opencv.org/)

</div>
