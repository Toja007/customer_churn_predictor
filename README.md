#                                                             Customer Churn Predictor Model
# Overview
This repository contains a customer churn predictor model developed using machine learning techniques. The model takes in customer information and predicts whether a customer will leave or not (churn prediction), providing valuable insights for customer retention strategies.

# Features Used
Customer_Age
Gender
Dependent_count
Education_Level
Marital_Status
Income_Category
Card_Category
Months_on_book
Total_Relationship_Count
Months_Inactive_12_mon
Contacts_Count_12_mon
Credit_Limit
Total_Revolving_Bal
Total_Amt_Chng_Q4_Q1
Total_Trans_Amt
Total_Trans_Ct
Total_Ct_Chng_Q4_Q1
Avg_Utilization_Ratio

# Model Selection
The customer churn predictor model was trained using various classifiers:

LogisticRegression
DecisionTreeClassifier
RandomForestClassifier
SVC
XGBClassifier
The XGBClassifier was found to be the best performing model for predicting customer churn

# Streamlit Application
A Streamlit application named "customer_churn_app.py" is included in this repository. You can run the Streamlit app locally to interactively input customer information and see the model's predictions for customer churn.
