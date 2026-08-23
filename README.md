# 📊 Customer Churn Prediction

An end-to-end **Customer Churn Prediction** machine learning project that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract information, payment method, tenure, and billing details.

The project includes **machine learning model development, preprocessing, model evaluation, explainability with SHAP, and a Flask web application** for interactive predictions.

---

## 🚀 Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are at higher risk of leaving allows businesses to take proactive retention actions.

This project uses customer information to:

* Predict whether a customer is likely to churn
* Estimate the probability of churn
* Display customer risk factors
* Highlight positive and negative factors influencing the prediction
* Provide an interactive web interface
* Visualize model performance and feature importance
* Use SHAP for model explainability

---

## ✨ Features

### 🤖 Machine Learning

* Customer churn classification
* Data preprocessing and feature transformation
* Saved trained model
* Saved preprocessing pipeline
* Model evaluation
* Feature importance analysis
* ROC curve analysis
* Confusion matrix
* SHAP-based explainability

### 🌐 Flask Web Application

The Flask application provides an interactive interface where users can enter customer information and receive a churn prediction.

The application considers information such as:

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
* Streaming TV
* Streaming movies
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

The application also provides customer-specific risk and positive factors based on the submitted information.

---

## 🧠 Machine Learning Workflow

```text
Customer Dataset
       │
       ▼
Data Cleaning
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
Saved Model + Preprocessor
       │
       ▼
Flask Web Application
       │
       ▼
Customer Churn Prediction
```

---

## 📁 Project Structure

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
└── templates/
```

### Important model files

`models/best_model.pkl` contains the trained prediction model.

`models/preprocessor.pkl` contains the preprocessing pipeline required to transform user input before prediction.

Both files are required when running the Flask application.

---

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **Gunicorn**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Matplotlib**
* **SHAP**
* **HTML / CSS**
* **Git & GitHub**
* **Render**

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/snehamahish45/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

Start the Flask application with:

```bash
python app.py
```

Or run it with Gunicorn:

```bash
gunicorn app:app
```

For Windows development, Flask can also be started using:

```bash
flask run
```

Then open the local address displayed by Flask in your browser.

---

## 🔮 Making a Prediction

1. Open the web application.
2. Enter the customer's information.
3. Submit the form.
4. The application processes the input using the saved preprocessor.
5. The trained model generates a churn prediction.
6. The application displays the prediction and relevant customer factors.

The application is designed to make the prediction process easy to understand rather than returning only a raw machine-learning output.

---

## 📈 Model Evaluation

The project includes several model evaluation outputs in the `models/` directory:

### Confusion Matrix

```text
models/confusion_matrix.png
```

Used to evaluate correct and incorrect classification results.

### ROC Curve

```text
models/roc_curve.png
```

Used to visualize the model's classification performance across different thresholds.

### Feature Importance

```text
models/feature_importance.csv
```

Contains feature-importance information from the trained model.

### Model Results

```text
models/model_results.csv
```

Contains model evaluation results.

### SHAP Summary

```text
models/shap_summary.png
```

Provides an explainability view of how features contribute to model predictions.

---

## 🔍 Explainable AI

This project uses **SHAP (SHapley Additive exPlanations)** to make machine-learning predictions easier to interpret.

Instead of simply saying:

> Customer is likely to churn.

the application can provide additional context around the customer's characteristics and identify factors associated with higher or lower churn risk.

Examples of risk factors considered by the application include:

* Short customer tenure
* Month-to-month contracts
* Higher monthly charges
* Electronic check payments
* Fiber-optic internet service

Examples of positive factors include:

* Longer tenure
* One-year or two-year contracts
* Automatic payment methods
* Higher predicted stay probability

These factors are intended as model/application insights and should not be interpreted as guaranteed causes of churn.

---

## 🌐 Deployment

The application can be deployed as a Flask web service on **Render**.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

The repository includes:

```text
runtime.txt
```

to specify the Python runtime used for deployment.

---

## ⚙️ Dependencies

The application currently uses the following main packages:

```text
Flask
Gunicorn
Pandas
NumPy
Scikit-learn
Joblib
Matplotlib
SHAP
```

Exact versions are maintained in:

```text
requirements.txt
```

---

## 🧪 Development

For development or experimentation, the project includes:

* `notebooks/` for exploratory/model-development work
* `src/` for project source code
* `models/` for trained models and evaluation outputs
* `templates/` for Flask HTML templates
* `static/` for frontend assets

---

## ⚠️ Important Notes

* The saved model and preprocessing files must remain available at:

  ```text
  models/best_model.pkl
  models/preprocessor.pkl
  ```
* The model should be used together with the preprocessing pipeline that was used during training.
* Predictions are model estimates and should not be treated as guaranteed outcomes.
* If the model is retrained, the corresponding preprocessing pipeline should also be updated.

---

## 🚀 Future Improvements

Possible future enhancements include:

* Hyperparameter optimization
* Model comparison and automated model selection
* Improved probability calibration
* Customer retention recommendations
* Interactive analytics dashboard
* Batch prediction for multiple customers
* Model monitoring
* Automated retraining pipeline
* API endpoint for programmatic predictions
* Improved mobile-friendly UI

---

## 👩‍💻 Author

**Sneha Mahish**

GitHub: `snehamahish45`

---

## 📄 License

This project is licensed under the terms specified in the repository's `LICENSE` file.

---

## ⭐ Project

If you find this project useful, consider giving the repository a star on GitHub.

**Customer Churn Prediction — Machine Learning + Flask + Explainable AI**
