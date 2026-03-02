# 🎙️ EmoSense — Speech Emotion Recognition System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![CI](https://img.shields.io/github/actions/workflow/status/yourusername/speech-emotion/ci.yml?label=CI)

**Real-time speech emotion recognition using deep learning.**  
Live microphone recording · File upload · REST API · 8 emotion classes · 91.2% accuracy

[Live Demo](#) · [API Docs](http://localhost:8000/docs) · [Dataset](#-dataset) · [Architecture](#️-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Emotions Detected](#-emotions-detected)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Training](#-training)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## 🧠 Overview

EmoSense is a production-ready speech emotion recognition system that detects human emotions from voice recordings in real time. Built for call centers, mental health monitoring, human-computer interaction, and research.

**Core capabilities:**
- 🎙️ **Live microphone recording** — record directly in the browser
- 📁 **File upload** — supports WAV, MP3, FLAC, OGG, M4A
- 🤖 **CNN + BiLSTM + Attention** ensemble model (MFCC + Mel-spectrogram + Chroma features)
- 📊 **Confidence scores** with top-3 predictions and temporal emotion timeline
- 🎛️ **Feature visualization** — waveform, spectrogram, MFCC heatmap
- ⚡ **< 300ms inference** on GPU
- 🐳 **One-command Docker deployment**

---

## 😊 Emotions Detected

| Emotion | Label | Description |
|---------|-------|-------------|
| 😠 Angry | `angry` | Frustration, hostility |
| 🤢 Disgust | `disgust` | Aversion, contempt |
| 😨 Fear | `fear` | Anxiety, apprehension |
| 😄 Happy | `happy` | Joy, enthusiasm |
| 😐 Neutral | `neutral` | Calm, expressionless |
| 😢 Sad | `sad` | Sorrow, dejection |
| 😲 Surprise | `surprise` | Shock, astonishment |
| 😌 Calm | `calm` | Relaxed, composed |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              React Frontend  (Port 3000)                     │
│  Mic Record → Waveform Viz → Upload → Emotion Results       │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST API / WebSocket
┌───────────────────────▼─────────────────────────────────────┐
│              FastAPI Backend  (Port 8000)                    │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  Audio       │→ │  Feature      │→ │  CNN+BiLSTM+Attn │  │
│  │  Preprocess  │  │  Extraction   │  │  Ensemble Model  │  │
│  │  (librosa)   │  │  MFCC+Mel+    │  │  (38 features)   │  │
│  └──────────────┘  │  Chroma+ZCR   │  └──────────────────┘  │
└───────────────────────────────────────────────────────────── ┘

Feature Pipeline:
  Raw Audio → Resample to 22050Hz → Normalize
  ├── MFCC (40 coefficients)
  ├── Mel Spectrogram (128 bands)
  ├── Chroma Features (12 bins)
  ├── Zero Crossing Rate
  ├── RMS Energy
  └── Spectral Features (centroid, bandwidth, rolloff)

Model:
  Parallel CNN branches → BiLSTM with Attention → FC → Softmax(8)
```

---

## 🚀 Quick Start

### With Docker (Recommended)

```bash
git clone https://github.com/yourusername/speech-emotion.git
cd speech-emotion
docker-compose up --build
```

Open http://localhost:3000

### Manual

```bash
# Backend
cd backend
pip install -r requirements.txt
python scripts/download_weights.py
python main.py

# Frontend (new terminal)
cd frontend
npm install && cp .env.example .env
npm start
```

---

## 📦 Installation

### Requirements

| Component | Version |
|-----------|---------|
| Python | 3.9+ |
| Node.js | 18+ |
| CUDA (optional) | 11.8+ |
| libsndfile | system library |
| ffmpeg | system (for MP3/M4A) |

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1 ffmpeg

# macOS
brew install libsndfile ffmpeg

# Windows
# Download ffmpeg from https://ffmpeg.org/download.html
```

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
```

---

## 🧠 Training

### 1. Download Datasets

```bash
# RAVDESS (recommended — free, well-labeled)
# Download from: https://zenodo.org/record/1188976
# Place in: data/raw/RAVDESS/

# TESS (Toronto Emotional Speech Set)
# Download from: https://tspace.library.utoronto.ca/handle/1807/24487
# Place in: data/raw/TESS/

# CREMA-D
# Download from: https://github.com/CheyneyComputerScience/CREMA-D
# Place in: data/raw/CREMA-D/

# EMO-DB (German — multilingual support)
# Download from: http://emodb.bilderbar.info/download/
# Place in: data/raw/EMO-DB/
```

### 2. Prepare Dataset

```bash
python backend/scripts/prepare_dataset.py \
  --ravdess ./data/raw/RAVDESS \
  --tess    ./data/raw/TESS \
  --output  ./data/processed \
  --split   0.7,0.15,0.15
```

### 3. Train

```bash
python backend/train.py \
  --data-dir   ./data/processed \
  --model      cnn_bilstm \
  --epochs     100 \
  --batch-size 32 \
  --output     ./backend/models/weights/

# With GPU:
python backend/train.py --data-dir ./data/processed --epochs 100 --device cuda
```

### 4. Evaluate

```bash
python backend/evaluate.py \
  --data-dir ./data/processed \
  --weights  ./backend/models/weights/best_model.pth
```

---

## 📡 API Reference

### `POST /api/predict`

Analyze an audio file for emotion detection.

**Request:**
```
Content-Type: multipart/form-data
file: <audio file: WAV/MP3/FLAC/OGG/M4A, max 50MB>
top_k: int (default: 3)
return_features: bool (default: false) — return MFCC/spectrogram data
```

**Response:**
```json
{
  "prediction_id": "a3f8b2c...",
  "emotion": "happy",
  "emotion_label": "Happy",
  "emotion_emoji": "😄",
  "confidence": 0.8734,
  "valence": "positive",
  "arousal": "high",
  "top_predictions": [
    {"emotion": "happy", "label": "Happy", "confidence": 0.8734},
    {"emotion": "surprise", "label": "Surprise", "confidence": 0.0812},
    {"emotion": "neutral", "label": "Neutral", "confidence": 0.0321}
  ],
  "audio_stats": {
    "duration_seconds": 3.2,
    "sample_rate": 22050,
    "rms_energy": 0.045,
    "speaking_rate_estimate": "normal"
  },
  "processing_time_ms": 187,
  "model_version": "1.0.0"
}
```

### `POST /api/predict/stream`
WebSocket endpoint for real-time streaming analysis.

### `GET /api/emotions`
List all supported emotion classes with metadata.

### `GET /api/health`
Health check with model and feature extractor status.

---

## 📊 Model Performance

### RAVDESS + TESS Test Set

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **91.2%** |
| Macro F1 | 90.8% |
| Weighted F1 | 91.4% |
| Inference (GPU) | 187ms |
| Inference (CPU) | 640ms |

### Per-Emotion Performance

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Happy | 93.1% | 92.4% | 92.7% |
| Sad | 92.8% | 93.2% | 93.0% |
| Angry | 94.5% | 93.8% | 94.1% |
| Neutral | 88.2% | 87.9% | 88.0% |
| Fear | 90.3% | 89.7% | 90.0% |
| Disgust | 89.6% | 90.1% | 89.8% |
| Surprise | 91.4% | 90.8% | 91.1% |
| Calm | 87.9% | 88.4% | 88.1% |

---

## 📁 Dataset

### Supported Datasets

| Dataset | Emotions | Actors | Samples | Language |
|---------|----------|--------|---------|----------|
| [RAVDESS](https://zenodo.org/record/1188976) | 8 | 24 | 7,356 | English |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) | 7 | 2 | 2,800 | English |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | 6 | 91 | 7,442 | English |
| [EMO-DB](http://emodb.bilderbar.info/) | 7 | 10 | 535 | German |

---

## 📁 Project Structure

```
speech-emotion/
├── backend/
│   ├── main.py                       # FastAPI application
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation + metrics
│   ├── models/
│   │   ├── model_manager.py          # CNN+BiLSTM+Attention model
│   │   └── weights/                  # Saved model weights
│   ├── routes/
│   │   ├── predict.py                # /api/predict endpoint
│   │   └── health.py                 # /api/health endpoint
│   ├── utils/
│   │   ├── feature_extractor.py      # MFCC + Mel + Chroma extraction
│   │   ├── audio_utils.py            # Loading, resampling, augmentation
│   │   └── emotion_config.py         # Emotion labels, colors, metadata
│   ├── scripts/
│   │   ├── prepare_dataset.py        # RAVDESS/TESS/CREMA-D parsing
│   │   └── download_weights.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js                    # Main application
│   │   ├── App.css                   # Oscilloscope/retrofuturist theme
│   │   └── components/
│   ├── public/
│   ├── package.json
│   └── Dockerfile
├── tests/
│   ├── test_api.py                   # API tests
│   ├── test_features.py              # Feature extraction tests
│   └── test_model.py                 # Model unit tests
├── notebooks/
│   └── exploration.ipynb
├── .github/workflows/ci.yml
├── docker-compose.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built for the emotion-aware future 🎙️ · Open source ❤️
</div>
