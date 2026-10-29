# ConnectTel Customer Churn Prediction

A machine learning project that analyzes customer churn patterns, compares classification models, and provides an interactive dashboard for predicting customer churn risk.

## 1. Project Overview

Customer churn occurs when customers discontinue a company's services. Predicting churn helps businesses identify customers who may leave and take proactive retention measures.

This project uses the Telco Customer Churn dataset to explore customer behavior, engineer features, train and evaluate machine learning models, interpret model predictions, and deploy a prediction API with an interactive Streamlit dashboard.

### Project Objectives

- Explore and understand customer churn patterns.
- Clean and preprocess customer data.
- Perform exploratory data analysis (EDA).
- Engineer meaningful features.
- Train and compare classification models.
- Evaluate models using classification metrics and ROC-AUC.
- Interpret important features using SHAP.
- Save the trained model for reuse.
- Deploy a Flask prediction API and a Streamlit dashboard.

## 2. Live Application

**Streamlit Dashboard:**  
https://customerchurnpredictor-auxawphrnssq5snhdyzvau.streamlit.app/

**Flask Prediction API:**  
https://customerchurnpredictor-nxr8.onrender.com

**API Health Check:**  
https://customerchurnpredictor-nxr8.onrender.com/health

The health endpoint returns the following response when the API is healthy and the model is loaded:

```json
{
  "model_loaded": true,
  "status": "healthy"
}
```

The dashboard communicates with the deployed Flask API to obtain churn predictions.

> Note: The API is hosted on Render. Depending on the hosting configuration, the service may take some time to respond after a period of inactivity.

## 3. Dataset

**Dataset:** Telco Customer Churn  
**Source:** Kaggle — IBM Telco Customer Churn dataset  
**Original records:** 7,043  
**Original columns:** 21

The dataset contains customer demographics, account information, subscribed services, billing details, and the target variable `Churn`.

### Data Cleaning

- Converted `TotalCharges` to numeric format.
- Identified 11 blank values in `TotalCharges`.
- Replaced those blank values with `0`, corresponding to customers with zero tenure.
- Checked the dataset for duplicate records; no duplicates were found.

## 4. Exploratory Data Analysis

EDA was used to investigate relationships between customer characteristics and churn.

The analysis included:

- Overall churn distribution.
- Churn by contract type.
- Churn by Internet service type.
- Churn by payment method.
- Customer tenure and churn.
- Monthly charges and total charges.
- Online security and technical support.
- Online backup and other subscribed services.

### Key Findings

- The overall churn rate was approximately **26.54%**.
- Month-to-month customers had a churn rate of **42.71%**.
- Customers on two-year contracts had a churn rate of **2.83%**.
- Customers using fiber-optic Internet had a churn rate of **41.89%**.
- Customers paying by electronic check had a churn rate of **45.29%**.
- The median tenure of churned customers was 10 months, compared with 38 months for customers who stayed.
- Customers who churned had higher average monthly charges: approximately **$74.44**, compared with **$61.27** for customers who stayed.
- Customers without online security or technical support showed comparatively high churn rates.

These are associations observed in the dataset; they do not establish that any individual factor causes churn.

## 5. Feature Engineering

Three additional features were created to represent customer service usage and billing patterns.

| Feature | Description |
|---|---|
| `TotalChargesPerTenure` | Total charges divided by tenure, with zero-tenure cases handled during feature engineering. |
| `ServiceCount` | Number of selected additional services marked as `Yes`. |
| `HasMultipleServices` | Binary indicator equal to `1` when the selected service count is at least two, and `0` otherwise. |

The three engineered features increased the feature set from 21 original columns to 24 columns before the target variable and preprocessing considerations are taken into account.

**Important:** The API input must use the feature names and data types expected by the saved model. Engineered features must be calculated consistently with the notebook's training logic.

## 6. Machine Learning Models

Three classification algorithms were trained and compared:

### Logistic Regression

A linear classification model used as a baseline. The model was configured with `max_iter=1000` and `class_weight="balanced"`.

### Random Forest Classifier

An ensemble model using 200 decision trees, with `class_weight="balanced"`.

### XGBoost Classifier

A gradient-boosted tree model configured with:

- `n_estimators=200`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`

The models were evaluated using a stratified 80:20 train-test split with `random_state=42`.

### Preprocessing

The preprocessing pipeline included:

- Median imputation for numerical features.
- Standard scaling for numerical features.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical features, with unknown categories ignored.

## 7. Model Evaluation Results

The following table preserves the results reported in the project.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 74.3% | 51.0% | **79.1%** | 0.842 |
| Random Forest | 75.9% | 54.1% | 61.2% | 0.822 |
| XGBoost | **80.6%** | **66.9%** | 53.5% | **0.845** |

### Interpretation

- **XGBoost** achieved the highest reported accuracy, precision, and ROC-AUC among these baseline models.
- **Logistic Regression** achieved the highest recall, making it worth considering when missing a customer who will churn is particularly costly.
- **Random Forest** performed between the other two models on several reported metrics but had the lowest ROC-AUC.

The preferred model depends on business priorities. A company focused on identifying as many potential churners as possible may prioritize recall, while a company seeking stronger overall ranking performance may prefer ROC-AUC.

### Logistic Regression Hyperparameter Tuning

Logistic Regression was also tuned using `GridSearchCV` with five-fold stratified cross-validation.

The search considered:

- `C`: `0.1`, `1`, and `10`
- `solver`: `liblinear` and `lbfgs`
- Scoring metric: ROC-AUC

The reported best cross-validation ROC-AUC was **0.8458**, while the reported test ROC-AUC was **0.841**. Tuning did not improve the reported test ROC-AUC over the baseline result.

## 8. Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to interpret the XGBoost model.

The analysis sampled up to 200 test observations and generated a SHAP summary plot displaying up to 15 features.

Important features reported in the analysis included:

- Contract type.
- Customer tenure.
- Online security.
- Internet service type.
- Payment method.
- Technical support.
- Monthly charges and total charges.

SHAP helps explain how features contribute to the model's predictions. These explanations describe model behavior and should not be interpreted as proof of causation.

## 9. Business Recommendations

Based on the patterns observed in the dataset, ConnectTel could consider the following actions:

1. **Focus on month-to-month customers:** Consider retention offers and incentives for longer-term contracts.
2. **Improve early-tenure engagement:** Monitor new customers closely and provide proactive onboarding support.
3. **Review electronic-check payment customers:** Investigate whether payment friction or other customer characteristics contribute to their higher observed churn rate.
4. **Evaluate service adoption:** Explore whether suitable online security, backup, and technical support packages improve customer retention.
5. **Monitor high-risk customers:** Use churn predictions to prioritize retention outreach.
6. **Choose prediction thresholds deliberately:** Evaluate precision, recall, intervention costs, and retention outcomes before adopting a threshold for real business decisions.

These recommendations should be validated through business experiments and additional analysis before implementation.

## 10. Project Architecture

The project contains a notebook for analysis, a saved model, a Flask API, and a Streamlit dashboard.

```text
CustomerChurnPredictor/
│
├── data/
│   └── Telco Customer Churn dataset
│
├── models/
│   └── churn_model.joblib
│
├── 01_Data_Exploration.ipynb
├── app.py
├── streamlit_app.py
├── Business_Insights_Report.md
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset is stored locally and excluded from Git. The saved model is stored under `models/`.

## 11. Technologies Used

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** Scikit-learn, XGBoost
- **Model explainability:** SHAP
- **Model persistence:** Joblib
- **Backend API:** Flask
- **Dashboard:** Streamlit
- **Production server:** Gunicorn
- **Version control:** Git and GitHub
- **API hosting:** Render
- **Dashboard hosting:** Streamlit Community Cloud
- **Development environment:** Jupyter Notebook and Visual Studio Code

## 12. Run the Project Locally

### Prerequisites

Install Python 3.12 or a compatible Python version supported by the project's dependencies. Git is recommended.

### Step 1: Clone the repository

```bash
git clone https://github.com/shrustipandey/CustomerChurnPredictor.git
cd CustomerChurnPredictor
```

### Step 2: Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify the model file

Ensure this file exists:

```text
models/churn_model.joblib
```

The prediction API requires the trained model file to be present. The dataset is not included in the public repository.

### Step 5: Start the Flask API

Open a terminal in the project root and run:

```bash
python app.py
```

If the Flask application is configured to run locally on port 5000, check:

```text
http://127.0.0.1:5000/health
```

For a production-style local run, Gunicorn can be used in an environment that supports it:

```bash
gunicorn app:app
```

### Step 6: Start the Streamlit dashboard

Open a second terminal in the project root, activate the same virtual environment, and run:

```bash
streamlit run streamlit_app.py
```

The dashboard should open in your browser. Ensure its configured API URL points to the intended API instance, either the deployed Render URL or your local Flask server.

## 13. Deployment

The project has been deployed as two components.

| Component | Platform | Purpose |
|---|---|---|
| Flask API | Render | Loads the trained model and serves prediction requests. |
| Streamlit dashboard | Streamlit Community Cloud | Provides the interactive user interface and sends requests to the API. |

The dashboard is configured to communicate with the deployed Render API.

### Deployment Verification

The API health endpoint has been tested and returned:

```json
{
  "model_loaded": true,
  "status": "healthy"
}
```

A churn prediction has also been obtained through the dashboard, confirming that the deployed interface and API can work together.

## 14. Limitations

- The reported results are based on a single stratified train-test split, except for the Logistic Regression cross-validation experiment.
- XGBoost's reported recall is 53.5%, meaning that the baseline model misses a substantial proportion of actual churners at the evaluated classification threshold.
- Additional cross-validation and hyperparameter tuning for the other models may improve the reliability of model comparisons.
- The model's performance may change on new customer populations or future data.
- SHAP explanations show associations used by the model, not causal relationships.
- Predictions should support, rather than replace, business judgment and should be validated before being used for real customer interventions.

## 15. Future Improvements

- Perform more comprehensive cross-validation across all candidate models.
- Tune XGBoost and Random Forest using a validation strategy that avoids selecting a model based on the final test set.
- Evaluate classification thresholds against retention campaign costs and benefits.
- Add monitoring for prediction quality and changes in customer behavior.
- Improve API input validation and automated testing.
- Expand the dashboard with more model explanations and retention insights.
- Evaluate the model on new data when it becomes available.

The Flask API and Streamlit dashboard are already deployed; future work focuses on improving evaluation, reliability, and functionality.

## 16. Author and Repository

**GitHub Repository:**  
https://github.com/shrustipandey/CustomerChurnPredictor

This project demonstrates an end-to-end machine learning workflow covering data analysis, feature engineering, model training, evaluation, explainability, API development, and deployment.
