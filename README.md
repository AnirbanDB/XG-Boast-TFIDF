# Firco XGBoost Compliance Predictor

> **Production-ready ML system for financial transaction compliance screening using TF-IDF vectorization and hierarchical XGBoost classification**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-8.0+-green.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Development](#development)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Firco XGBoost Compliance Predictor** is an enterprise-grade machine learning system designed for financial institutions to automate compliance decision-making for transaction screening alerts. The system processes MT103 and MT202 SWIFT messages, analyzing watchlist hits to predict compliance actions.

### Business Problem
Financial institutions receive thousands of transaction screening alerts daily. Manual review is time-consuming and expensive. This system automates the decision-making process with 87%+ accuracy.

### Solution
A hierarchical ML pipeline that predicts 4 compliance targets:
- **Hit-level decisions**: Review decisions and comments
- **Message-level decisions**: Final actions and reviewer comments

---

## ✨ Key Features

### Machine Learning
- **Hierarchical XGBoost Classification** with 4 target predictions
- **TF-IDF Text Vectorization** with optimized n-gram features (1-2 grams, 2000 features)
- **Advanced Feature Engineering**: 2,500+ features across text, categorical, and numerical data
- **Cross-Validation** with early stopping to prevent overfitting
- **Class Balancing** for handling imbalanced datasets
- **Model Versioning** with automatic archiving and metadata tracking

### API & Architecture
- **FastAPI REST API** with 15+ endpoints
- **Async/Await** patterns for non-blocking operations
- **Background Task Management** for long-running ML training
- **Real-time & Batch Predictions**
- **MongoDB Integration** for audit trails and state persistence
- **AWS S3 Support** for model storage (optional)
- **Docker Containerization** for easy deployment

### Production Features
- **Comprehensive Error Handling** with detailed logging
- **Health Check Endpoints** for monitoring
- **Performance Tracking** with metrics storage
- **Model Validation** with automated performance reports
- **File Upload Support** for CSV batch processing
- **CORS Enabled** for web integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Endpoints │  │ Background   │  │ State        │       │
│  │   (15+)     │  │ Tasks        │  │ Manager      │       │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
│   ML Pipeline   │ │   MongoDB    │ │   AWS S3     │
│                 │ │              │ │  (Optional)  │
│ • Data Load     │ │ • Audit      │ │ • Backups    │
│ • Preprocessing │ │ • Metadata   │ │ • Models     │
│ • TF-IDF        │ │ • State      │ │              │
│ • XGBoost       │ │              │ │              │
│ • Evaluation    │ │              │ │              │
└─────────────────┘ └──────────────┘ └──────────────┘
```

### Component Architecture

```
models/
├── base_model.py           # Abstract base for all models
├── tfidf_xgb_F.py         # Hierarchical XGBoost implementation
└── __init__.py            # Model factory pattern

Core Modules:
├── xgb_app_F.py           # FastAPI application (entry point)
├── main_xgb_F.py          # Training & prediction pipeline
├── config.py              # Centralized configuration
├── dataset_utils.py       # Data preprocessing
├── train_utils.py         # Training utilities
├── api_utils.py           # API helpers
├── async_tasks.py         # Background task manager
├── state_manager.py       # Database-driven state
├── mongo_utils.py         # MongoDB operations
├── s3_utils.py            # AWS S3 integration
└── schemas.py             # Pydantic validation schemas
```

---

## 🛠️ Technology Stack

### Core ML Stack
- **Python 3.9+** - Programming language
- **XGBoost 2.0+** - Gradient boosting framework
- **scikit-learn 1.3+** - ML pipeline & TF-IDF vectorization
- **Pandas 2.0+** - Data manipulation
- **NumPy 1.24+** - Numerical computing

### API & Backend
- **FastAPI 0.104+** - Modern async web framework
- **Uvicorn 0.24+** - ASGI server
- **Pydantic 2.0+** - Data validation
- **Motor 3.3+** - Async MongoDB driver
- **PyMongo 4.5+** - MongoDB sync driver

### Storage & Infrastructure
- **MongoDB 8.0+** - Document database
- **AWS S3** - Object storage (optional)
- **Docker** - Containerization

### Development Tools
- **python-dotenv** - Environment management
- **joblib** - Model serialization
- **pytest** - Testing framework

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- MongoDB 8.0+ (local or Atlas)
- pip package manager
- (Optional) Docker for containerized deployment

### Step 1: Clone the Repository
```bash
git clone https://github.com/AnirbanDB/XG-Boast-TFIDF.git
cd XG-Boast-TFIDF/Firco/xgb
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up MongoDB

#### Option A: Local MongoDB
```bash
# macOS
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Ubuntu/Debian
sudo apt-get install mongodb
sudo systemctl start mongodb

# Verify installation
mongosh --eval "db.version()"
```

#### Option B: MongoDB Atlas (Cloud)
1. Create free account at https://www.mongodb.com/cloud/atlas
2. Create cluster and database user
3. Whitelist your IP address
4. Get connection string

### Step 5: Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `Firco/xgb/` directory:

```bash
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=firco_local

# AWS S3 Configuration (Optional)
# AWS_ACCESS_KEY_ID=your_access_key_here
# AWS_SECRET_ACCESS_KEY=your_secret_key_here
# AWS_S3_BUCKET_NAME=your_bucket_name
# AWS_S3_REGION=us-east-1
```

### Configuration Files

Key configuration in `config.py`:

```python
# Model directories
MODEL_SAVE_DIR = "./saved_models"
ARCHIVE_DIR = "./archive"
UPLOADS_DIR = "./uploads"
PREDICTIONS_DIR = "./predictions"

# Target columns (4 predictions)
HIT_LEVEL_TARGETS = ["hit.review_decision", "hit.review_comments"]
MESSAGE_LEVEL_TARGETS = ["decision.last_action", "decision.reviewer_comments"]

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# TF-IDF parameters
TEXT_PROCESSING = {
    'max_features': 2000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.9
}
```

---

## 🚀 Usage

### Starting the API Server

#### Development Mode
```bash
# Navigate to project directory
cd Firco/xgb

# Start with auto-reload
uvicorn xgb_app_F:app --host 0.0.0.0 --port 3004 --reload
```

#### Production Mode
```bash
# Start with optimized settings
uvicorn xgb_app_F:app --host 0.0.0.0 --port 3004 --workers 4
```

The API will be available at:
- **API**: http://localhost:3004
- **Swagger Docs**: http://localhost:3004/docs
- **ReDoc**: http://localhost:3004/redoc

### Training a Model

#### Using API (Recommended)
```bash
curl -X POST "http://localhost:3004/v1/firco-xgb/train" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@firco_alerts_final_5000_7.csv" \
  -F "user_id=your_user_id" \
  -F "model_name=compliance_model_v1"
```

#### Using Python Script
```python
from main_xgb_F import train_model_without_save

# Train model
model, results, label_encoders = train_model_without_save(
    csv_file="firco_alerts_final_5000_7.csv",
    test_size=0.2,
    val_size=0.1
)

print(f"Overall Accuracy: {results['overall_metrics']['accuracy']:.2%}")
```

### Making Predictions

#### Single Prediction (API)
```bash
curl -X POST "http://localhost:3004/v1/firco-xgb/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "hit.matching_text": "JOHN DOE",
      "hit.watchlist_text": "SANCTIONS LIST",
      "hit.score": 95.5,
      "hit.is_pep": "false",
      "mt103.country": "US"
    },
    "user_id": "your_user_id"
  }'
```

#### Batch Prediction (CSV Upload)
```bash
curl -X POST "http://localhost:3004/v1/firco-xgb/predict-batch" \
  -F "file=@test_data.csv" \
  -F "user_id=your_user_id"
```

#### Python Script
```python
from main_xgb_F import predict_single_input, load_model

# Load model
model = load_model("saved_models/v1.pkl")

# Single prediction
input_data = {
    "hit.matching_text": "JOHN DOE",
    "hit.score": 95.5,
    # ... other features
}

predictions = predict_single_input(model, input_data)
print(predictions)
```

### Model Validation
```bash
curl -X POST "http://localhost:3004/v1/firco-xgb/validate" \
  -F "file=@validation_data.csv" \
  -F "user_id=your_user_id"
```

---

## 📚 API Documentation

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint with API info |
| `GET` | `/v1/firco-xgb/health` | Health check status |
| `GET` | `/v1/firco-xgb/training-status` | Check training progress |
| `POST` | `/v1/firco-xgb/train` | Train new model |
| `POST` | `/v1/firco-xgb/predict` | Single prediction |
| `POST` | `/v1/firco-xgb/predict-batch` | Batch predictions |
| `POST` | `/v1/firco-xgb/validate` | Validate model |
| `GET` | `/v1/firco-xgb/models` | List all models |
| `GET` | `/v1/firco-xgb/models/{version}` | Get model details |
| `GET` | `/v1/firco-xgb/models/{version}/download` | Download model |
| `GET` | `/v1/firco-xgb/models/{version}/performance` | Model metrics |
| `GET` | `/v1/firco-xgb/feature-importance` | Feature importance analysis |
| `POST` | `/v1/firco-xgb/upload-csv` | Upload CSV file |
| `GET` | `/v1/firco-xgb/predictions/{filename}` | Download predictions |

### Interactive Documentation

Once the server is running, access comprehensive interactive documentation:

- **Swagger UI**: http://localhost:3004/docs
  - Try out endpoints directly
  - View request/response schemas
  - Generate code snippets

- **ReDoc**: http://localhost:3004/redoc
  - Clean, three-panel design
  - Detailed descriptions
  - Export to OpenAPI spec

---

## 📊 Model Performance

### Overall Metrics
- **Validation Accuracy**: 87.09%
- **Overall F1 Score**: 84.26%
- **Training Samples**: ~15,000
- **Validation Samples**: ~1,800
- **Test Samples**: ~200

### Target-Specific Performance

| Target | Accuracy | F1 Score | Description |
|--------|----------|----------|-------------|
| `hit.review_decision` | 77.94% | 72.15% | Hit-level review decision |
| `hit.review_comments` | 95.64% | 94.85% | Hit-level comments |
| `decision.last_action` | 77.06% | 71.42% | Message-level action |
| `decision.reviewer_comments` | 97.71% | 97.18% | Message-level comments |

### Feature Importance (Top 10)
1. `hit.score` - Match confidence score
2. `hit.matching_text_tfidf_0` - TF-IDF features from matching text
3. `hit.watchlist_text_tfidf_0` - TF-IDF features from watchlist
4. `hit.is_pep` - PEP (Politically Exposed Person) flag
5. `mt103.country` - Transaction origin country
6. `hit.hit_type` - Type of screening hit
7. `decision.last_action` - Previous decision (for hierarchical predictions)
8. `mt202.amount` - Transaction amount
9. `hit.matching_text_tfidf_1` - Additional TF-IDF features
10. `mt103.hits_count_103` - Number of hits in MT103

---

## 📁 Project Structure

```
Firco/xgb/
├── README.md                      # This file
├── .env                           # Environment configuration (git-ignored)
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker build configuration
├── docker-compose.yml             # Docker Compose setup
│
├── xgb_app_F.py                   # FastAPI application (MAIN ENTRY)
├── main_xgb_F.py                  # Training & prediction pipeline
├── config.py                      # Configuration management
│
├── models/                        # ML model implementations
│   ├── __init__.py                # Model factory
│   ├── base_model.py              # Abstract base model
│   └── tfidf_xgb_F.py             # XGBoost implementation
│
├── api_utils.py                   # API utility functions
├── async_tasks.py                 # Background task manager
├── dataset_utils.py               # Data preprocessing
├── train_utils.py                 # Training utilities
├── state_manager.py               # Application state management
├── mongo_utils.py                 # MongoDB operations
├── s3_utils.py                    # AWS S3 integration
├── database.py                    # Database connection
├── crud.py                        # Database CRUD operations
├── schemas.py                     # Pydantic schemas
│
├── saved_models/                  # Trained model artifacts
│   ├── v1.pkl                     # Model version 1
│   ├── v1_results.json            # Training results
│   └── ...
│
├── archive/                       # Archived models
├── splits/                        # Train/val/test splits
├── uploads/                       # Uploaded CSV files
├── predictions/                   # Prediction outputs
│
├── test_api_F.py                  # API tests
├── test_batch_prediction.py       # Batch prediction tests
├── debug_prediction_endpoint.py   # Debug utilities
└── demo_base_model.py             # Model demonstrations
```

---

## 🔧 Development

### Running Tests
```bash
# Run API tests
python test_api_F.py

# Run batch prediction tests
python test_batch_prediction.py

# Run with pytest (if configured)
pytest tests/ -v
```

### Code Quality
```bash
# Format code with black
black .

# Lint with flake8
flake8 . --max-line-length=120

# Type checking with mypy
mypy . --ignore-missing-imports
```

### Debugging
```bash
# Debug prediction endpoint
python debug_prediction_endpoint.py

# Check vectorizer
python debug_vectorizer.py

# Demo base model
python demo_base_model.py
```

### Creating Custom Data Splits
```python
# Run custom split creation
python create_custom_splits.py
```

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
# Build image
docker build -t firco-xgb:latest .

# Run container
docker run -d \
  -p 3004:3004 \
  -e MONGODB_URI=mongodb://host.docker.internal:27017/ \
  -e MONGODB_DATABASE=firco_local \
  --name firco-xgb-api \
  firco-xgb:latest
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Compose Configuration
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "3004:3004"
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/
      - MONGODB_DATABASE=firco_local
    depends_on:
      - mongodb
  
  mongodb:
    image: mongo:8.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed
- Keep commits atomic and descriptive

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Anirban Deb**
- GitHub: [@AnirbanDB](https://github.com/AnirbanDB)
- Repository: [XG-Boast-TFIDF](https://github.com/AnirbanDB/XG-Boast-TFIDF)

---

## 🙏 Acknowledgments

- **XGBoost Team** - For the excellent gradient boosting framework
- **FastAPI Team** - For the modern async web framework
- **scikit-learn Team** - For comprehensive ML tools
- **MongoDB Team** - For robust document database

---

## 📞 Support

For questions, issues, or feature requests:
- **GitHub Issues**: https://github.com/AnirbanDB/XG-Boast-TFIDF/issues
- **Documentation**: http://localhost:3004/docs (when server is running)

---

## 🗺️ Roadmap

- [ ] Add support for additional message types (MT950, MT940)
- [ ] Implement model A/B testing framework
- [ ] Add Prometheus metrics for monitoring
- [ ] Implement model explainability (SHAP values)
- [ ] Add GraphQL API support
- [ ] Implement streaming predictions with Kafka
- [ ] Add multi-language support for UI
- [ ] Kubernetes deployment configurations

---

**Made with ❤️ for Financial Compliance Automation**
