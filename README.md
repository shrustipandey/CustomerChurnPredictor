# Customer Churn Prediction — ConnectTel

## 1. Project Overview

This project predicts whether a telecom customer is likely to leave the company (churn). It uses machine learning to identify important churn patterns and support customer-retention strategies.

## 2. Objectives

- Explore customer data and identify churn patterns.
- Perform data cleaning and feature engineering.
- Train and compare Logistic Regression, Random Forest, and XGBoost.
- Evaluate models using Accuracy, Precision, Recall, and ROC-AUC.
- Use SHAP to understand important prediction features.
- Save the trained model for future use.

## 3. Dataset

The project uses the Telco Customer Churn dataset from Kaggle.

Expected CSV location:

`data/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains 7,043 customer records. Blank values in `TotalCharges` were converted to numeric values; 11 records with zero tenure had their missing `TotalCharges` set to 0.

## 4. Feature Engineering

Three additional features were created:

- `TotalChargesPerTenure`
- `ServiceCount`
- `HasMultipleServices`

## 5. Models and Test Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.743 | 0.510 | 0.791 | 0.842 |
| Random Forest | 0.759 | 0.541 | 0.612 | 0.822 |
| XGBoost | 0.806 | 0.669 | 0.535 | 0.845 |

XGBoost achieved the highest ROC-AUC among these tested models. Logistic Regression achieved higher recall, which may be useful when identifying as many potential churners as possible.

## 6. Model Interpretation

SHAP was used to examine which features influence XGBoost predictions. Important features included contract type, tenure, online security, internet service type, payment method, and technical support.

Feature importance indicates associations in model predictions; it does not establish causation.

## 7. Project Structure

- `01_Data_Exploration.ipynb` — data exploration, preprocessing, feature engineering, model training, evaluation, and SHAP analysis.
- `data/archive/` — source dataset.
- `models/churn_model.joblib` — saved XGBoost pipeline.
- `Business_Insights_Report.md` — business findings and recommendations.
- `requirements.txt` — Python package dependencies.
- `README.md` — project documentation.

## 8. Setup and Usage

Use Python 3.12 or a compatible Python environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `01_Data_Exploration.ipynb` and run the cells in order.

## 9. Limitations

- Results depend on the dataset and its representativeness.
- Model performance may change with new customer data.
- A high churn-risk prediction does not guarantee that a customer will leave.
- This project is currently a local machine-learning project; a public prediction API and deployed web interface are separate deployment steps.

## 10. Future Improvements

- Tune and validate models more extensively.
- Evaluate business costs associated with missed churners and false alarms.
- Build a prediction API and interactive web interface.
- Containerize and deploy the application.

## 11. Disclaimer

This project is for educational and analytical purposes. Predictions should support, not replace, business judgment.
