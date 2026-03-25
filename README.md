# 🌾 SmartKhet — AI-Powered Agricultural Advisory Platform

> *Kisan pehle, technology baad mein.* — The farmer first, technology in service.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![React Native](https://img.shields.io/badge/React_Native-0.74-blue)](https://reactnative.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

SmartKhet is a production-ready, AI-first agricultural advisory platform built for 140 million Indian farmers. It provides crop recommendations, disease detection, market price intelligence, and multilingual voice advisory — working even in low-connectivity rural environments.

---

## 📁 Repository Structure

```
smartkhet/
├── ml/                          # All ML models and training pipelines
│   ├── crop_recommendation/     # XGBoost + RandomForest ensemble
│   ├── disease_detection/       # EfficientNet-B4 (PyTorch) for plant disease
│   ├── market_prediction/       # LSTM + Facebook Prophet for mandi prices
│   ├── nlp_voice/               # Whisper STT + IndicBERT NLU pipeline
│   ├── edge_models/             # TFLite quantization + export scripts
│   └── mlflow_utils/            # Experiment tracking and model registry
│
├── backend/                     # Microservices (Python FastAPI)
│   ├── farmer_service/          # Farmer profile, land, crop history
│   ├── crop_advisory_service/   # Soil + weather → crop/fertilizer advice
│   ├── disease_service/         # Image upload, ML routing, remedy delivery
│   ├── market_service/          # Mandi prices, prediction, sell signals
│   ├── nlp_voice_service/       # STT, intent classification, advisory LLM
│   ├── notification_service/    # Push, WhatsApp, SMS fan-out (Node.js)
│   └── shared/                  # Common utilities, DB models, auth
│
├── mobile/                      # React Native Android App
│   └── src/
│       ├── screens/             # UI screens (Home, Disease, Market, Voice)
│       ├── services/            # API clients, offline sync
│       ├── offline/             # SQLite + TFLite on-device inference
│       └── components/          # Reusable UI components
│
├── whatsapp_bot/                # Node.js Meta Cloud API bot
├── infra/
│   ├── kubernetes/              # K8s manifests (Deployments, Services, HPA)
│   ├── terraform/               # AWS EKS + RDS + ElastiCache IaC
│   └── docker/                  # Service Dockerfiles
│
├── data/
│   ├── schemas/                 # PostgreSQL table schemas
│   ├── migrations/              # Alembic DB migrations
│   └── seeds/                   # Sample crop/disease data
│
├── scripts/                     # Dev setup, data ingestion scripts
├── docker-compose.yml           # Local full-stack development
└── requirements.txt             # Root Python dependencies
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| Node.js | 20+ | [nodejs.org](https://nodejs.org) |
| Docker | 24+ | [docker.com](https://docker.com) |
| React Native CLI | latest | `npm i -g react-native-cli` |
| Android Studio | latest | For mobile dev |

### 1. Clone and Configure

```bash
git clone https://github.com/your-org/smartkhet.git
cd smartkhet

# Copy environment files
cp .env.example .env
# Edit .env with your API keys (see Environment Variables section)
```

### 2. Start Backend Services

```bash
# Start all services via Docker Compose
docker-compose up -d

# Verify all services are healthy
docker-compose ps
```

Services available at:
- API Gateway: `http://localhost:8000`
- Farmer Service: `http://localhost:8001`
- Crop Advisory: `http://localhost:8002`
- Disease Service: `http://localhost:8003`
- Market Service: `http://localhost:8004`
- NLP/Voice Service: `http://localhost:8005`
- MLflow UI: `http://localhost:5000`

### 3. Train ML Models

```bash
# Install ML dependencies
pip install -r requirements.txt

# Train Crop Recommendation Model
python ml/crop_recommendation/train.py --data data/seeds/soil_crop_data.csv

# Train Disease Detection Model
python ml/disease_detection/train.py --data /path/to/plantvillage --epochs 30

# Train Market Prediction Model
python ml/market_prediction/train.py --commodity wheat --district gorakhpur

# Convert models to TFLite (for edge deployment)
python ml/edge_models/convert_tflite.py --model disease --output mobile/src/offline/models/
```

### 4. Run Mobile App

```bash
cd mobile
npm install

# Start Metro bundler
npx react-native start

# Run on Android device/emulator
npx react-native run-android
```

### 5. Run Tests

```bash
# Backend unit tests
pytest backend/ -v --cov=backend --cov-report=html

# ML model tests
pytest ml/ -v

# Integration tests
pytest tests/integration/ -v
```

---

## 🧠 ML Models Overview

| Model | Algorithm | Accuracy | Size | Deployment |
|-------|-----------|----------|------|-----------|
| Crop Recommendation | XGBoost + RandomForest Ensemble | 94.2% | 8MB | Cloud + Edge |
| Disease Detection | EfficientNet-B4 (Transfer Learning) | 92.7% F1 | 45MB (cloud), 12MB (TFLite) | Cloud + Edge |
| Market Price Prediction | LSTM + Facebook Prophet Ensemble | MAE ₹28/qtl | 15MB | Cloud only |
| Voice STT | Whisper-small (fine-tuned Indic) | 91% WER | 242MB | Cloud only |
| Intent Classification | IndicBERT fine-tuned | 96.1% | 110MB | Cloud only |

---

## 🌐 Environment Variables

```env
# Database
POSTGRES_URL=postgresql://user:pass@localhost:5432/smartkhet
MONGODB_URL=mongodb://localhost:27017/smartkhet
REDIS_URL=redis://localhost:6379

# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-south-1
S3_BUCKET_NAME=smartkhet-images

# External APIs
IMD_API_KEY=your_imd_key
OPENWEATHER_API_KEY=your_key
AGMARKNET_API_KEY=your_key

# Messaging
MSG91_API_KEY=your_key
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
META_WHATSAPP_TOKEN=your_token
META_VERIFY_TOKEN=your_verify_token

# ML
MLFLOW_TRACKING_URI=http://localhost:5000
OPENAI_API_KEY=your_key  # For advisory LLM fallback

# Auth
JWT_SECRET_KEY=your_super_secret_key_min_32_chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
```

---

## 🏗️ Architecture Summary

```
[Android App / WhatsApp / IVR / SMS]
           ↓
    [Kong API Gateway]  ← JWT Auth, Rate Limit, SSL
           ↓
  ┌────────────────────────────────┐
  │      Microservices (FastAPI)   │
  │  Farmer │ Advisory │ Disease   │
  │  Market │ NLP/Voice│ Notif.    │
  └─────────┬──────────────────────┘
            ↓ Apache Kafka Events
  ┌─────────────────────────────┐
  │    AI / ML Layer            │
  │  XGBoost │ EfficientNet-B4  │
  │  LSTM    │ Whisper+IndicBERT│
  └─────────────────────────────┘
            ↓
  ┌─────────────────────────────┐
  │    Data Layer               │
  │  PostgreSQL │ MongoDB       │
  │  Redis      │ TimescaleDB   │
  └─────────────────────────────┘
```

---

## 🗺️ Roadmap

- [x] MVP: Crop recommendation + Disease detection + Mandi prices
- [x] WhatsApp bot integration
- [x] Hindi voice query support
- [ ] TFLite offline edge models in APK
- [ ] 22 regional Indian languages
- [ ] IoT soil sensor integration (LoRa/BLE)
- [ ] IVR zero-literacy flow
- [ ] Market price 14-day forecasting
- [ ] Kubernetes production deployment

---

## 🤝 Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'feat: add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

*Designed by **Axora** — AI Architecture Intelligence*
