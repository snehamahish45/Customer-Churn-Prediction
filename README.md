# Customer Churn Prediction

<p align="center">
  <b>End-to-End Machine Learning Application for Predicting Customer Churn</b>
</p>

<p align="center">
  <a href="https://github.com/snehamahish45/Customer-Churn-Prediction">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-purple?style=for-the-badge" alt="SHAP">
</p>

---

## Overview

**Customer Churn Prediction** is an end-to-end machine learning project designed to identify customers who are at risk of leaving a subscription-based service.

The project combines:

* Data preprocessing
* Machine learning classification
* Model evaluation
* Feature importance analysis
* SHAP-based explainability
* Flask web application
* Production deployment with Gunicorn

The final application allows users to enter customer information through a web interface and receive a churn prediction along with interpretable insights.

---

## Key Features

### Machine Learning

* Customer churn classification
* Automated preprocessing pipeline
* Trained production model
* Model evaluation and comparison
* Confusion matrix
* ROC curve
* Feature importance analysis
* SHAP explainability

### Web Application

* Interactive customer input form
* Real-time churn prediction
* Churn probability
* Customer risk assessment
* Positive and negative prediction factors
* Responsive web interface

### Deployment

* Flask production application
* Gunicorn WSGI server
* Python runtime configuration
* Deployment-ready model artifacts
* Compatible with cloud platforms such as Render

---

## Application Workflow

```text
                    Customer Information
                            │
                            ▼
                  ┌─────────────────────┐
                  │   Flask Web Form     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Preprocessor     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Trained Model     │
                  └──────────┬──────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          Churn Prediction       Churn Probability
                  │                     │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Explainable Insights│
                  │      / SHAP         │
                  └─────────────────────┘
```

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── LICENSE
│
├── data/
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   ├── model_results.csv
│   ├── roc_curve.png
│   └── shap_summary.png
│
├── notebooks/
│
├── src/
│
├── static/
│
├── templates/
│
└── catboost_info/
```

### Production Model Artifacts

The application uses two important serialized artifacts:

| File                      | Purpose                        |
| ------------------------- | ------------------------------ |
| `models/best_model.pkl`   | Trained machine learning model |
| `models/preprocessor.pkl` | Feature preprocessing pipeline |

Both artifacts must be available for the Flask application to perform predictions.

---

## Technology Stack

| Category             | Technology      |
| -------------------- | --------------- |
| Programming Language | Python          |
| Web Framework        | Flask           |
| Production Server    | Gunicorn        |
| Data Processing      | Pandas, NumPy   |
| Machine Learning     | Scikit-learn    |
| Model Persistence    | Joblib / Pickle |
| Visualization        | Matplotlib      |
| Explainable AI       | SHAP            |
| Frontend             | HTML, CSS       |
| Version Control      | Git, GitHub     |
| Deployment           | Render          |

---

## Machine Learning Pipeline

The project follows a standard supervised machine learning workflow:

```text
Data
  │
  ▼
Data Cleaning
  │
  ▼
Exploratory Data Analysis
  │
  ▼
Feature Engineering
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
Best Model
  │
  ▼
Model Serialization
  │
  ▼
Flask Application
```

---

## Model Evaluation

The repository includes model evaluation artifacts under the `models/` directory.

### Confusion Matrix

`models/confusion_matrix.png`

Provides a visual breakdown of correct and incorrect classifications.

### ROC Curve

`models/roc_curve.png`

Shows the model's classification performance across different decision thresholds.

### Feature Importance

`models/feature_importance.csv`

Contains feature-importance information generated from the trained model.

### Model Comparison

`models/model_results.csv`

Contains model evaluation results used to compare candidate models.

### SHAP Summary

`models/shap_summary.png`

Provides an explainability view of the features influencing model predictions.

---

## Explainable AI

A key component of this project is **model interpretability**.

SHAP is used to understand how individual features contribute to model predictions.

Instead of providing only:

```text
Prediction: Customer likely to churn
```

the application can provide additional context around the factors contributing to the prediction.

This makes the model more useful for analysis and customer-retention decision support.

> **Note:** Model explanations describe learned relationships in the training data. They should not automatically be interpreted as causal relationships.

---

## Customer Information

The prediction interface uses customer attributes such as:

* Gender
* Senior citizen status
* Partner
* Dependents
* Tenure
* Phone service
* Multiple lines
* Internet service
* Online security
* Online backup
* Device protection
* Tech support
* Streaming services
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/snehamahish45/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The Flask application will start locally.

Open the local URL displayed in the terminal to access the application.

---

## Production Deployment

The application is configured for deployment using Gunicorn.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

The repository also includes `runtime.txt` to define the Python runtime used by the deployment environment.

---

## Configuration

The application expects the following model files:

```text
models/best_model.pkl
models/preprocessor.pkl
```

If either file is missing, the application cannot initialize the prediction pipeline.

---

## Results & Insights

The project provides a complete prediction and interpretation workflow rather than treating machine learning as a black box.

The generated artifacts allow analysis of:

* Classification performance
* False positives and false negatives
* Discriminative performance through ROC analysis
* Relative feature importance
* Individual feature contributions through SHAP

---

## Future Improvements

Potential improvements include:

* Hyperparameter optimization
* Cross-validation and automated model selection
* Probability calibration
* Customer segmentation
* Batch prediction support
* REST API endpoints
* Interactive analytics dashboard
* Model monitoring
* Automated retraining
* CI/CD pipeline
* Improved accessibility and mobile responsiveness

---

## Disclaimer

This project is intended for educational, analytical, and demonstration purposes.

Predictions generated by the model are estimates based on patterns learned from the training data and should not be treated as guaranteed outcomes.

---

## Author

### Sneha Mahish

GitHub:
https://github.com/snehamahish45

---

## License

This project is available under the license specified in the repository's `LICENSE` file.

---

## Repository

**Customer Churn Prediction**

https://github.com/snehamahish45/Customer-Churn-Prediction

If you find this project useful, consider giving the repository a ⭐.
