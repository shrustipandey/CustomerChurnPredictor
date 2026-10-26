# ConnectTel — Customer Churn Prediction
## Business Insights and Model Evaluation Report

### 1. Executive Summary

This project analyzes customer churn for ConnectTel using machine-learning classification models. The objective is to identify customers at risk of leaving and provide recommendations to improve retention.

The dataset contains 7,043 customer records. Three engineered features were added to the 21 original columns, resulting in 24 columns.

Three algorithms were compared: Logistic Regression, Random Forest, and XGBoost. XGBoost achieved the highest test ROC-AUC (0.845), accuracy (80.6%), and precision (66.9%). Logistic Regression achieved the highest recall (79.1%), which may be preferable when identifying as many potential churners as possible is the priority.

### 2. Dataset Overview and Preparation

- Total customers: 7,043
- Customers who stayed: 5,174 (73.46%)
- Customers who churned: 1,869 (26.54%)
- Duplicate rows detected: 0
- The original `TotalCharges` column contained 11 blank entries.

The 11 blank `TotalCharges` entries were converted to numeric missing values and replaced with zero because all corresponding customers had zero recorded tenure. This assumption should be documented when interpreting the results.

Three engineered features were added:

1. **TotalChargesPerTenure:** Total charges divided by tenure. Zero-tenure cases were left missing for preprocessing.
2. **ServiceCount:** Number of selected additional services used by each customer.
3. **HasMultipleServices:** Indicates whether a customer uses at least two of the selected additional services.

The dataset was split into training and testing sets using an 80:20 ratio, with stratification to preserve the churn distribution.

### 3. Key Exploratory Data Analysis Findings

#### Contract Type
- Month-to-month customers: 42.71% churn.
- One-year contract customers: 11.27% churn.
- Two-year contract customers: 2.83% churn.

Month-to-month customers are a high-priority group for further retention analysis.

#### Internet Service
- Fiber-optic customers: 41.89% churn.
- DSL customers: 18.96% churn.
- Customers without internet service: 7.40% churn.

The higher churn rate among fiber-optic customers warrants investigation into pricing, service quality, support, and customer expectations.

#### Payment Method
- Electronic check: 45.29% churn.
- Mailed check: 19.11% churn.
- Automatic bank transfer: 16.71% churn.
- Automatic credit card: 15.24% churn.

Electronic-check customers are a key segment for further investigation.

#### Tenure
- Median tenure among retained customers: 38 months.
- Median tenure among churned customers: 10 months.
- Average tenure among retained customers: 37.57 months.
- Average tenure among churned customers: 17.98 months.

Customers who churned generally had shorter tenure, suggesting early customer engagement may be important.

#### Monthly Charges
- Median monthly charges among retained customers: $64.425.
- Median monthly charges among churned customers: $79.650.
- Average monthly charges among retained customers: $61.27.
- Average monthly charges among churned customers: $74.44.

Higher monthly charges were associated with churn in this dataset. Pricing and perceived value should be investigated further.

#### Additional Services
Higher churn rates were observed among customers without:
- Online security: 41.77% churn.
- Tech support: 41.64% churn.
- Online backup: 39.93% churn.

These relationships may help prioritize further investigation, but they do not establish that the absence of these services causes churn.

### 4. Machine-Learning Model Comparison

The models were evaluated on the same held-out test set.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 74.3% | 51.0% | 79.1% | 0.842 |
| Random Forest | 75.9% | 54.1% | 61.2% | 0.822 |
| XGBoost | 80.6% | 66.9% | 53.5% | 0.845 |

XGBoost achieved the highest test accuracy, precision, and ROC-AUC among the three baseline models. Its ROC-AUC of 0.845 indicates useful ability to distinguish between customers who churn and those who stay.

Logistic Regression achieved the highest recall at 79.1%, identifying a larger proportion of actual churners at the default classification threshold.

The tuned Logistic Regression model achieved a five-fold cross-validation ROC-AUC of 0.8458 and a test ROC-AUC of 0.841. Tuning did not improve its test ROC-AUC over the original baseline.

### 5. Explainable AI — SHAP Analysis

SHAP was used to examine how features influenced the XGBoost model's predictions. The summary plot highlighted influential features including:

- Contract type, particularly month-to-month contracts
- Customer tenure
- Online security
- Internet service type
- Payment method
- Tech support
- Total charges and monthly charges

SHAP helps explain model behavior. These explanations are not proof that any feature causes churn.

### 6. Business Recommendations

**1. Prioritize month-to-month customers.** Investigate retention offers, contract flexibility, onboarding, and customer satisfaction for this high-churn segment.

**2. Investigate electronic-check customers.** Examine whether this segment differs in contract type, tenure, service mix, or customer experience before designing interventions.

**3. Review fiber-optic customer experience.** Investigate service quality, pricing, technical issues, and support satisfaction among fiber-optic customers.

**4. Strengthen early-tenure engagement.** Consider proactive follow-ups, onboarding support, and early identification of dissatisfied customers.

**5. Review optional service adoption.** Test whether suitable online security, backup, and technical support bundles improve customer satisfaction and retention.

**6. Use risk predictions carefully.** Evaluate the cost of missed churners versus unnecessary retention offers before choosing a model and classification threshold.

### 7. Limitations

- The findings are based on historical customer data and do not prove causation.
- The reported metrics are based on one held-out test split.
- The baseline XGBoost model has recall of 53.5%, so it misses a substantial proportion of actual churners at its current classification threshold.
- Cross-validation and further tuning of the remaining models should be completed for a stronger comparison.
- SHAP explains model behavior rather than establishing business causes.
- Retention recommendations should be tested using controlled experiments where practical.

### 8. Conclusion

The analysis identified several customer segments associated with higher churn, including month-to-month contract customers, fiber-optic customers, electronic-check customers, and customers with shorter tenure.

XGBoost achieved the highest test ROC-AUC among the three baseline models, while Logistic Regression achieved higher recall. The final model choice should depend on ConnectTel's retention objectives and the relative cost of missed churners versus unnecessary interventions.

The results provide a foundation for targeted retention strategies and further model validation before operational deployment.
