# Loan Default Prediction - End-to-End MLOps Project

An end-to-end **production-grade machine learning system** that predicts whether a loan will be **paid back or defaulted**. This project demonstrates the complete ML lifecycle from experimentation to deployment, featuring **reproducible ML pipelines**, **experiment tracking**, **version control**, and **automated CI/CD**.

---

## 🚀 Project Highlights

* Built **XGBoost classification model** achieving **92.1% cross-validated AUC** and **87% accuracy**
* Implemented **reproducible ML pipeline** with **DVC** for data versioning and orchestration
* Tracked **20+ parameters** and **15+ metrics** across experiments using **MLflow**
* Collaborated and versioned models with **DagShub** (remote storage + experiment tracking)
* Achieved **89% recall** on positive class using **optimal threshold selection** (Youden's J-statistic)
* Deployed model as **FastAPI REST API** with **Streamlit UI**, fully **Dockerized**
* Automated **CI/CD pipeline** with **GitHub Actions** (testing, building, deployment)
* Handled **class imbalance** using `scale_pos_weight` and **Optuna** hyperparameter optimization

---

## 🧠 ML Model & Pipeline

### Model Architecture
* **Algorithm:** XGBoost (Gradient Boosted Trees)
* **Problem Type:** Binary Classification
  * `0` → Loan Default
  * `1` → Loan Paid Back

### ML Pipeline (DVC-Orchestrated)
```
1. data_ingestion      → Load train.csv, test.csv
2. data_preprocessing  → Feature engineering, ordinal encoding for grades
3. model_training      → Optuna tuning (10 trials, 3-fold CV) + XGBoost training
4. model_evaluation    → Optimal threshold selection + classification report
```

### Key Techniques
* **Optuna** hyperparameter optimization (3-fold stratified CV on full training set)
* **Ordinal encoding** for credit grade hierarchy (A1 < A2 < ... < G5)
* **Youden's J-statistic** for optimal decision threshold (0.478)
* **Class imbalance handling** with `scale_pos_weight`
* **No data leakage**: Validation split AFTER Optuna tuning

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Optuna 3-Fold CV AUC** | **0.921** |
| **Validation AUC** | **0.921** |
| **Accuracy** | **87%** |
| **Precision (Class 1)** | **94%** |
| **Recall (Class 1)** | **89%** |
| **F1-Score (Class 1)** | **92%** |
| **Optimal Threshold** | **0.478** |

> 📈 **Training Data**: 593,994 samples | **Validation**: 118,799 samples (20% split)

---

## 🛠️ MLOps Stack

### **Experiment Tracking & Versioning**
* **MLflow**: Experiment tracking, parameter/metric logging, model registry
* **DVC**: Data versioning, pipeline orchestration, reproducibility
* **DagShub**: Remote experiment tracking, team collaboration, S3 storage backend

### **Deployment & Infrastructure**
* **FastAPI**: REST API for real-time inference
* **Streamlit**: Interactive web UI for predictions
* **Docker**: Containerized API and UI services
* **Docker Compose**: Multi-container orchestration

### **CI/CD & Automation**
* **GitHub Actions**: Automated testing, linting, Docker builds
* **Docker Hub**: Automated image publishing (khushbu308/loan-api)
* **pytest**: Unit tests for API endpoints and model validation

---

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Development Phase                         │
├─────────────────────────────────────────────────────────────┤
│  DVC Pipeline → MLflow Tracking → DagShub Collaboration     │
│  (data versioning)  (experiments)    (remote storage)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  GitHub Push → Tests (pytest) → Docker Build → Docker Hub   │
│                (lint, unit test)  (automated)   (publish)    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Production Deployment                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User (Browser) → Streamlit UI (Docker) → FastAPI (Docker)  │
│                    Port 8501              Port 8000          │
│                                              ↓               │
│                                        XGBoost Model         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure
```
LOAN_PREDICTION_API/
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # CI pipeline (testing, building)
│       └── cd.yml              # CD pipeline (deployment)
│
├── api/                        # FastAPI backend
│   ├── model/
│   │   ├── model.pkl          # Trained XGBoost model
│   │   ├── model_metadata.json
│   │   └── predict.py         # Prediction logic
│   ├── schema/
│   │   ├── user_input.py
│   │   └── prediction_response.py
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── ui/                         # Streamlit frontend
│   ├── frontend.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── src/                        # ML pipeline scripts
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── tests/                      # Unit tests
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── data/
│   ├── raw/                    # DVC-tracked datasets
│   └── processed/              # Pipeline outputs
│
├── model/                      # Training artifacts
│   ├── model.pkl              # DVC-tracked
│   └── model_metadata.json
│
├── mlruns/                     # MLflow experiments
├── reports/                    # Evaluation reports
├── notebooks/                  # EDA & experimentation
│
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Hyperparameters & config
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Root dependencies
└── README.md
```

---

## 🔄 ML Pipeline Workflow (DVC)

### **Pipeline Stages**
```bash
# Run complete pipeline
dvc repro

# Individual stages:
# 1. Data Ingestion
python src/data_ingestion.py

# 2. Preprocessing
python src/data_preprocessing.py

# 3. Training (Optuna + XGBoost)
python src/model_training.py

# 4. Evaluation (Threshold + Metrics)
python src/model_evaluation.py
```

### **Pipeline DAG**
```
data/raw/train.csv
        ↓
   data_ingestion
        ↓
data/processed/X_train.csv
        ↓
  data_preprocessing
        ↓
   model_training (Optuna + XGBoost)
        ↓
   model_evaluation (Threshold + Report)
        ↓
   model/model.pkl
```

### **Reproducibility**
```bash
# Version data with DVC
dvc add data/raw/train.csv
dvc push

# Version model
dvc add model/model.pkl
dvc push

# Reproduce entire pipeline
dvc repro
```

---

## 📊 Experiment Tracking (MLflow + DagShub)

### **What Gets Logged**

**Parameters (20+):**
```python
n_trials, cv_folds, val_size, random_state,
n_estimators, max_depth, learning_rate,
subsample, colsample_bytree, min_child_weight,
gamma, reg_alpha, reg_lambda, ...
```

**Metrics (15+):**
```python
optuna_cv_auc, val_auc, training_time,
optimal_threshold, tpr_at_optimal, fpr_at_optimal,
accuracy, precision_class_0, precision_class_1,
recall_class_0, recall_class_1, f1_class_0, f1_class_1
```

**Artifacts:**
```
- model/model.pkl (sklearn pipeline)
- model_metadata.json
- classification_report.json
```

### **MLflow UI**
```bash
# Start MLflow server
mlflow ui --port 5000

# View at: http://localhost:5000
```

### **DagShub Integration**

All experiments synced to DagShub for:
- Team collaboration
- Remote experiment tracking
- Model versioning
- S3-backed storage

---

## 🚀 CI/CD Pipeline

### **Continuous Integration (CI)**

**Triggers:** Every push to `main`

**Steps:**
1. ✅ **Lint code** (flake8, black, isort)
2. ✅ **Run tests** (pytest with coverage)
3. ✅ **Build Docker images** (API + UI)
4. ✅ **Test containers** (health checks)

### **Continuous Deployment (CD)**

**Triggers:** Push to `main` branch or version tags

**Steps:**
1. ✅ **Build production images**
2. ✅ **Tag with version** (latest, v2.0, commit-sha)
3. ✅ **Push to Docker Hub**
4. ✅ **(Optional) Deploy to EC2**

### **GitHub Actions Workflows**
```yaml
.github/workflows/
├── ci.yml    # Testing + building
└── cd.yml    # Deployment to Docker Hub
```

**Automated Checks:**
- Unit tests (API endpoints, model loading)
- Docker build validation
- Container health checks
- Code quality (linting, formatting)

---

## 🐳 Docker Deployment

### **Docker Images**

* **API**: `khushbu308/loan-api:latest`
* **UI**: `khushbu308/loan-ui:latest`


**Test Coverage:**
- API endpoints (health, predict)
- Model loading and inference
- Data preprocessing
- Schema validation

---

## 🎯 Key Technical Achievements

### **ML Engineering**
* ✅ Reproducible pipeline with DVC
* ✅ Experiment tracking with MLflow
* ✅ Hyperparameter optimization with Optuna
* ✅ Optimal threshold selection (Youden's J)
* ✅ Class imbalance handling

### **MLOps**
* ✅ Automated CI/CD (GitHub Actions)
* ✅ Containerized deployment (Docker)
* ✅ Model versioning (DVC + MLflow)
* ✅ Remote collaboration (DagShub)

### **Software Engineering**
* ✅ REST API (FastAPI)
* ✅ Interactive UI (Streamlit)
* ✅ Unit testing (pytest)
* ✅ Code quality (linting, formatting)

---

## 📚 Technologies Used

| Category | Tools |
|----------|-------|
| **ML/DL** | XGBoost, scikit-learn, Optuna |
| **MLOps** | MLflow, DVC, DagShub |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Data** | pandas, NumPy |
| **DevOps** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest, pytest-cov |
| **Cloud** | Docker Hub, AWS S3 (DVC remote) |

---


## 👤 Author

**Khushbu**  
MSc Mathematics | ML & MLOps Engineer

---

⭐ **If you found this project useful, please star the repository!**

📚 **For questions or collaboration, feel free to open an issue or reach out!**