# Loan Default Prediction (End-to-End ML Deployment)

An end-to-end **machine learning + deployment** project that predicts whether a loan will be **paid back or defaulted**. The project covers the full lifecycle: data preprocessing, model training with **XGBoost**, evaluation and threshold tuning, API serving with **FastAPI**, interactive UI with **Streamlit**, and complete **Docker & Docker Compose**–based deployment.

---

## 🚀 Project Highlights

* Built an **XGBoost-based loan default classification model**
* Achieved **ROC–AUC ≈ 0.92** on validation data
* Handled **class imbalance** using:

  * `scale_pos_weight`

* Selected optimal probability threshold using ROC curve
* Deployed model as a **FastAPI inference service**
* Built an interactive **Streamlit UI** for predictions
* Fully **Dockerized** backend & frontend
* Orchestrated services using **Docker Compose**
* Debugged and resolved real-world Docker issues (networking, ports, images, containers)

---

## 🧠 ML Model Details

* **Algorithm:** XGBoost (Gradient Boosted Trees)
* **Problem Type:** Binary Classification

  * `0` → Loan Default
  * `1` → Loan Paid Back
* **Key Techniques Used:**

  * Feature preprocessing & encoding
  * Handling class imbalance with `scale_pos_weight`
  * Threshold tuning instead of relying on default `0.5`

### 📊 Performance Snapshot

| Metric                        | Value      |
| ----------------------------- | ---------- |
| Accuracy                      | 0.86       |
| ROC–AUC                       | 0.92       |
| Cross-Validation ROC-AUC      | 0.92       |


> Focused on probability prediction and threshold tuning rather than relying on default 0.5 classification.

---

## 🏗️ System Architecture

```
User (Browser)
   │
   ▼
Streamlit UI (Docker)
   │  
   ▼
FastAPI API (Docker)
   │
   ▼
XGBoost Model
```

* Containers communicate via **Docker internal network**
* Streamlit connects to FastAPI using service name (`api:8000`)

---

## 🧩 Tech Stack

* **Language:** Python 3.11
* **ML:** XGBoost, scikit-learn, pandas, numpy
* **Backend API:** FastAPI + Uvicorn
* **Frontend:** Streamlit
* **Containerization:** Docker
* **Orchestration:** Docker Compose

---

## 📁 Project Structure

```
loan-project/
│
├── api/                    # FastAPI backend
│   ├── data
│   ├── model
    |   ├──  model.pkl
    │   ├── predict.py
│   ├── schema
    |   ├──  user_input.py
    |   ├──  prediction_response.py
│   ├── app.py
│   ├── Predicting_loan_payback.ipynb
│   ├── requirements.txt
│   └── Dockerfile
│
├── ui/                     # Streamlit frontend
│   ├── frontend.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── .dockerignore
├── .gitignore
└── docker-compose.yml
```

---

## 🐳 Docker & Deployment

### Docker Images

* **FastAPI (Model API):** `khushbu308/loan-api`
* **Streamlit UI:** `khushbu308/loan-prediction-ui`

Both images are pushed to **Docker Hub**.

### Run with Docker Compose

```bash
docker-compose up
```

This will start:

* FastAPI on `http://localhost:8000`
* Streamlit UI on `http://localhost:8501`

---

## 🔍 Docker Debugging (Real-World Issues Solved)

During deployment, multiple real-world Docker issues were debugged and resolved:

* Port conflicts (`Bind for 0.0.0.0:8000 failed`)
* Mixing `docker run` with `docker-compose`
* Container-to-container networking (`localhost` vs service name)
* Missing executables (`uvicorn not found`)
* Incorrect build vs image usage in `docker-compose.yml`
* Stale containers and orphaned networks

> These debugging steps reflect **production-grade DevOps troubleshooting**, not toy examples.

---

## 🧪 API Usage Example

### Endpoint

```
POST /predict
```

### Sample Request

```json
{
  "annual_income": 30000,
  "debt_to_income_ratio": 0.16,
  "credit_score": 636,
  "loan_amount": 5000,
  "interest_rate": 12.9,
  "gender": "Male",
  "marital_status": "Married",
  "education_level": "High School",
  "employment_status": "Employed",
  "loan_purpose": "Other",
  "grade_subgrade": "A3"
}
```

---

## 🎯 Key Learnings

* Accuracy alone is misleading for imbalanced problems
* Threshold tuning can outperform model re-training
* Docker networking requires **service-name-based communication**
* Clean container lifecycle management is critical
* End-to-end ML projects require **ML + backend + DevOps** skills

---

## 📌 Future Improvements

* Model versioning
* CI/CD pipeline (GitHub Actions)
* Deployment to AWS EC2 / ECS
* Monitoring & logging
* Authentication layer for API

---

## 👤 Author

**Khushbu**
MSc Mathematics | ML & AI Enthusiast

---

⭐ If you found this project useful, feel free to star the repository!
