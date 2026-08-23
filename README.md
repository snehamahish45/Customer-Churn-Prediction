# Customer Churn Prediction

### End-to-End Machine Learning | Explainable AI | Flask | Production Deployment

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.9-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-6A5ACD?style=flat-square" />
  <img src="https://img.shields.io/badge/Gunicorn-Production-499848?style=flat-square&logo=gunicorn&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployment-Render-46E3B7?style=flat-square&logo=render&logoColor=111111" />
</p>

<p align="center">
  <a href="https://github.com/snehamahish45/Customer-Churn-Prediction">Repository</a>
  •
  <a href="#installation">Installation</a>
  •
  <a href="#deployment">Deployment</a>
  •
  <a href="#model-interpretability">Explainability</a>
</p>

---

## Overview

Customer churn is a critical business problem for subscription-based organizations. Retaining an existing customer is often more efficient than acquiring a new one, making early identification of potential churners an important analytical task.

This project develops an **end-to-end customer churn prediction system** that combines machine learning, explainable AI, and a production-ready Flask application.

The system takes customer attributes as input, applies the same preprocessing pipeline used during model development, generates a churn prediction, and presents interpretable insights that help explain the prediction.

### What this project demonstrates

* End-to-end supervised machine learning
* Data preprocessing and feature transformation
* Classification model development
* Model evaluation and comparison
* Feature importance analysis
* SHAP-based model explainability
* Model serialization and inference
* Flask application development
* Production serving with Gunicorn
* Cloud deployment with Render
* Reproducible Python environment

---

# Business Objective

The objective is to identify customers who have a higher likelihood of churn so that businesses can prioritize retention strategies.

The model can support questions such as:

> **Which customers are most likely to churn?**

> **How strong is the predicted churn risk?**

> **Which customer characteristics contribute most to the prediction?**

The application is designed as a **decision-support tool**, allowing model predictions to be combined with business knowledge when evaluating customer-retention strategies.

---

# Solution Architecture

```text
                         ┌──────────────────────┐
                         │   Customer Inputs     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Flask Interface    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Input Validation    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Preprocessor      │
                         │  preprocessor.pkl     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Trained ML Model    │
                         │    best_model.pkl     │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         ▼                      ▼
                ┌─────────────────┐    ┌──────────────────┐
                │ Churn Prediction│    │ Churn Probability│
                └────────┬────────┘    └────────┬─────────┘
                         │                      │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Explainable Insights  │
                         │       / SHAP          │
                         └──────────────────────┘
```

---

# Machine Learning Pipeline

The project follows a structured machine learning workflow:

```text
Raw Data
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Preparation
   │
   ▼
Preprocessing
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ├── Confusion Matrix
   ├── ROC Curve
   ├── Feature Importance
   └── SHAP Analysis
   │
   ▼
Best Performing Model
   │
   ▼
Model Serialization
   │
   ▼
Flask Inference Application
   │
   ▼
Production Deployment
```

---

# Dataset & Features

The prediction interface uses customer-level attributes representing demographic, service, contract, and billing information.

### Customer Profile

* Gender
* Senior citizen status
* Partner
* Dependents
* Tenure

### Services

* Phone service
* Multiple lines
* Internet service
* Online security
* Online backup
* Device protection
* Tech support
* Streaming TV
* Streaming movies

### Contract & Billing

* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

These variables provide a combination of behavioral, contractual, service, and financial information for predicting churn.

---

# Model Development

The project separates **data transformation** from **model inference** through a persisted preprocessing pipeline.

This ensures that customer inputs submitted through the Flask application are transformed consistently with the data used during model training.

### Production artifacts

| Artifact           | Purpose                            |
| ------------------ | ---------------------------------- |
| `best_model.pkl`   | Persisted trained prediction model |
| `preprocessor.pkl` | Persisted preprocessing pipeline   |

The application loads these artifacts during startup and uses them for inference.

---

# Model Evaluation

The repository contains several evaluation artifacts used to assess model performance.

### Confusion Matrix

```text
models/confusion_matrix.png
```

Evaluates classification outcomes including correct predictions and classification errors.

### ROC Curve

```text
models/roc_curve.png
```

Visualizes the model's ability to distinguish between churn and non-churn customers across classification thresholds.

### Feature Importance

```text
models/feature_importance.csv
```

Provides feature-level importance information generated during model evaluation.

### Model Comparison

```text
models/model_results.csv
```

Stores evaluation results used during model comparison.

### SHAP Summary

```text
models/shap_summary.png
```

Provides a global view of how features influence model predictions.

---

# Model Interpretability

## Explainable AI with SHAP

A production machine learning system should not only produce predictions; it should also provide insight into **why** those predictions were generated.

This project incorporates SHAP-based explainability to analyze the contribution of individual features to model predictions.

The explainability layer helps answer:

* Which features increase predicted churn risk?
* Which features are associated with lower predicted churn risk?
* Which variables have the strongest influence on model output?
* How does an individual customer's profile affect the prediction?

This makes the system more transparent and useful for analytical decision-making.

> **Important:** Feature contributions represent model behavior and learned statistical relationships. They should not automatically be interpreted as causal effects.

---

# Web Application

The Flask application provides a user-facing prediction interface.

### Prediction workflow

```text
1. Enter customer information
              ↓
2. Submit prediction request
              ↓
3. Validate input
              ↓
4. Apply saved preprocessing pipeline
              ↓
5. Generate model prediction
              ↓
6. Calculate prediction probability
              ↓
7. Generate explanatory insights
              ↓
8. Display results
```

The interface is designed to translate machine-learning output into information that is easier for a non-technical user to understand.

---

# Technology Stack

| Layer             | Technology      |
| ----------------- | --------------- |
| Language          | Python          |
| Web Framework     | Flask           |
| ML Framework      | Scikit-learn    |
| Data Processing   | Pandas, NumPy   |
| Model Persistence | Joblib / Pickle |
| Explainability    | SHAP            |
| Visualization     | Matplotlib      |
| Production Server | Gunicorn        |
| Frontend          | HTML, CSS       |
| Version Control   | Git / GitHub    |
| Deployment        | Render          |

---

# Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py                         # Flask application
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python runtime configuration
├── README.md
├── LICENSE
│
├── data/                          # Dataset / project data
│
├── models/
│   ├── best_model.pkl             # Production model
│   ├── preprocessor.pkl           # Preprocessing pipeline
│   ├── confusion_matrix.png       # Evaluation artifact
│   ├── feature_importance.csv     # Feature importance
│   ├── model_results.csv          # Model evaluation results
│   ├── roc_curve.png              # ROC visualization
│   └── shap_summary.png            # SHAP visualization
│
├── notebooks/                     # Analysis and experimentation
│
├── src/                           # Supporting source code
│
├── static/                        # CSS / static assets
│
└── templates/                     # Flask HTML templates
```

---

# Installation

## Prerequisites

* Python 3.12+
* Git
* pip

## Clone the repository

```bash
git clone https://github.com/snehamahish45/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

## Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running Locally

Start the Flask application:

```bash
python app.py
```

For production-style execution:

```bash
gunicorn app:app
```

Then open the local URL provided by Flask or your deployment environment.

---

# Deployment

The application is configured for production deployment using Gunicorn.

### Build command

```bash
pip install -r requirements.txt
```

### Start command

```bash
gunicorn app:app
```

The repository also contains `runtime.txt` to define the Python runtime used by the deployment platform.

### Production requirements

The deployment environment must contain:

```text
models/best_model.pkl
models/preprocessor.pkl
```

Without these artifacts, the Flask application cannot initialize the prediction pipeline.

---

# Reproducibility

The project maintains its Python dependencies in:

```text
requirements.txt
```

The Python runtime is specified through:

```text
runtime.txt
```

This helps keep local development and deployment environments consistent.

When retraining the model, the preprocessing pipeline should be regenerated and saved together with the model to ensure compatibility during inference.

---

# Engineering Considerations

The project follows several practical ML engineering principles:

### Consistent preprocessing

Training and inference use the same persisted preprocessing pipeline.

### Separation of artifacts

The trained model and preprocessing logic are stored independently, making the inference workflow explicit.

### Explainability

Model predictions are complemented by feature-level interpretation.

### Production serving

The Flask application is served through Gunicorn rather than relying on Flask's development server for production workloads.

### Versioned dependencies

Package versions are maintained in `requirements.txt` to improve reproducibility.

---

# Future Roadmap

Potential production enhancements include:

* Automated hyperparameter optimization
* Cross-validation and experiment tracking
* Probability calibration
* Model monitoring and drift detection
* Automated retraining
* REST API endpoints
* Batch prediction
* Customer segmentation
* Retention recommendation engine
* CI/CD automation
* Unit and integration testing
* Containerized deployment
* Performance monitoring
* Interactive business analytics dashboard

---

# Limitations

The model's predictions are dependent on the quality and representativeness of the training data.

Potential limitations include:

* Historical data may not represent future customer behavior.
* Model performance may vary across customer segments.
* Feature relationships may change over time.
* Predictions indicate statistical patterns rather than guaranteed outcomes.
* Explainability results describe model behavior and should not automatically be interpreted as causal relationships.

---

# Intended Use

This project is intended for:

* Machine learning portfolio demonstration
* Educational purposes
* Customer analytics
* Predictive modeling research
* Explainable AI demonstration
* Flask ML application development

It should not be used as the sole basis for consequential business decisions without appropriate validation and human review.

---

# Author

## Sneha Mahish

Machine Learning | Data Science | Python

**GitHub:**
https://github.com/snehamahish45

---

# License

This project is distributed under the license specified in the repository's `LICENSE` file.

---

<p align="center">
  <b>Customer Churn Prediction</b><br>
  Machine Learning • Explainable AI • Flask • Production Deployment
</p>

<p align="center">
  ⭐ If you find this project useful, consider starring the repository.
</p>
