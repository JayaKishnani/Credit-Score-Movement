# Credit Score Movement Prediction
This project simulates a realistic financial dataset and builds a machine learning model to predict credit score movement (increase, decrease, or stable) based on a variety of customer financial behaviors. It also incorporates model explainability tools like <code>SHAP</code> to interpret model decisions.

## 📌 Objective
- Simulate a customer-level dataset with features related to income, credit behavior, and loan history.

- Train a multi-class classification model to predict the movement of credit scores.

- Handle class imbalance using resampling techniques.

- Use model explainability tools to highlight key drivers of predictions.

- Visualize insights and explore data using various EDA techniques.

## 🧾 Dataset Description

| Feature                        | Distribution Used         | Description                                        | Reason                                                            |
| ------------------------------ | ------------------------- | -------------------------------------------------- | ----------------------------------------------------------------- |
| `customer_id`                  | Random Integer            | Unique customer identifier                         | To simulate uniqueness of each record                             |
| `age`                          | Random Integer            | Age of customer (21–65)                            | To cover a realistic credit-active demographic                    |
| `gender`                       | Random Choice             | Male/Female                                        | To reflect binary gender representation                           |
| `location`                     | Random Choice             | City of residence                                  | Represents geographic diversity                                   |
| `monthly_income`               | Normal                    | Monthly income (₹20,000 – ₹200,000)                | Continuous, centered around an average                            |
| `monthly_emi_outflow`          | Ratio of income           | EMI paid monthly                                   | Realistic percentage of income (10%–60%)                          |
| `current_outstanding`          | Normal                    | Total outstanding credit                           | Debt levels follow bell curves with upper limits                  |
| `credit_utilization_ratio`     | Beta (scaled)             | Ratio of credit used to total limit                | Bounded ratio, skewed toward responsible users                    |
| `num_open_loans`               | Poisson                   | Number of open loans                               | Count of loans – event frequency model                            |
| `repayment_history_score`      | Normal                    | Score based on repayment history (0–100)           | Scores typically follow a bell curve                              |
| `dpd_last_3_months`            | Exponential               | Days past due over last 3 months                   | Time/days till payment — skewed toward early payers               |
| `num_hard_inquiries_last_6m`   | Poisson                   | Number of hard inquiries in last 6 months          | Count of hard inquiries                                           |
| `recent_credit_card_usage`     | Normal                    | Credit card usage in recent month                  | Spending varies around an average                                 |
| `recent_loan_disbursed_amount` | Custom Choices            | Recent loan amount disbursed                       | Discrete disbursement amounts, with many zeros                    |
| `total_credit_limit`           | Derived                   | Estimated total credit limit                       | Based on outstanding / utilization ratio                          |
| `months_since_last_default`    | Random Choice             | Months since last default                          | Mixture of zeros (recent defaulters) and various durations        |
| `target_credit_score_movement` | Rule-based Classification | Target variable (`increase`, `decrease`, `stable`) | Derived using domain logic on EMI/inquiries/utilization/repayment |


### Labeling Logic
The credit score movement is determined based on a heuristic involving DPD, credit utilization, EMI-to-income ratio, inquiries, and repayment score.

## 📊 Exploratory Data Analysis
- Distribution plots of target classes.

- Scatter plots of financial variables.

- Violin plots to visualize feature distributions across classes.

- Heatmap to analyze the interaction between gender and credit movement.

## 🛠️ Modeling Approach
### Preprocessing:

- Label encoding for categorical features (gender, location).

- Balanced the dataset using upsampling for minority classes.

### Model:

- Trained a Random Forest Classifier.

- Evaluated using confusion matrix and classification report.

## 🧠 Model Explainability
### SHAP (SHapley Additive exPlanations)
Global explanation method to understand feature importance across the test set. Uses TreeExplainer for interpreting the Random Forest model.

<img alt = "shap_image" src = "">

#### Key Insights:
🔹 `dpd_last_3_months` is the most influential feature, strongly contributing to Class 0 (credit score decrease), but also affects all classes.

🔸 Other top contributors include:

`repayment_history_score`

`num_hard_inquiries_last_6m`

`monthly_emi_outflow`

`credit_utilization_ratio`

🔻 Features like `gender`, `num_open_loans`, and `current_outstanding` have minimal impact on predictions and could potentially be dropped or deprioritized.

## Segment Based Interventions

| Segment              | Indicators (SHAP)                        | Risks/Opportunities   | Suggested Actions                   |
| -------------------- | ---------------------------------------- | --------------------- | ----------------------------------- |
| **High-Risk**        | High `dpd`, high `inquiries`, high `emi` | Default, churn        | Payment plans, freeze rules, alerts |
| **High-Opportunity** | Good `history_score`, low `dpd`          | Upsell, build loyalty | Credit upgrades, personalized loans |
| **Stable**           | Middle-ground SHAP impact                | Improve engagement    | Nudges, score-based education       |


## 📁 Files
- `credit_score_movement_dataset.csv`: Simulated dataset.

- `rf_model`: Serialized Random Forest model (via joblib).

- `credit_score_movement.ipynb`: Main notebook with all code (data generation, EDA, modeling, explainability).