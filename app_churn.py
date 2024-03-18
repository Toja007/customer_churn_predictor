import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = pickle.load(open("churn_model.pkl", 'rb'))

# Label Encoder for categorical features
le = LabelEncoder()

# Function to encode categorical variables
def encode_categorical(df):
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Education_Level'] = le.fit_transform(df['Education_Level'])
    df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
    df['Income_Category'] = le.fit_transform(df['Income_Category'])
    df['Card_Category'] = le.fit_transform(df['Card_Category'])
    return df

# Function to predict churn
def predict_churn(customer_data):
    #Encode categorical variables
    customer_data_encoded = encode_categorical(customer_data)
    
    # Predict churn and get probability
    prediction = model.predict(customer_data_encoded)
    probability = model.predict_proba(customer_data_encoded)[0][1]
    return prediction, probability

# Main function for Streamlit app
def main():
    st.title('Customer Churn Prediction App')

    # Input form for customer information
    st.sidebar.header('Input Customer Information')
    customer_age = st.sidebar.slider('Customer Age', 18, 100, 30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    dependent_count = st.sidebar.slider('Dependent Count', 0, 10, 0)
    education_level = st.sidebar.selectbox('Education Level', {'High School': 3, 'Graduate': 2, 'Uneducated': 5, 'Unknown': 6, 'College': 0, 'Post-Graduate': 4, 'Doctorate': 1})
    marital_status = st.sidebar.selectbox('Marital Status', {'Married': 1, 'Single': 2, 'Unknown': 3})
    income_category = st.sidebar.selectbox('Income Category', {'Less than $40K': 4, '$60K - $80K': 2, '$80K - $120K': 3, '$40K - $60K': 1, 'Unknown': 5, '$120K +': 0})
    card_category = st.sidebar.selectbox('Card Category', {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3})
    months_on_book = st.sidebar.slider('Months on Book', 1, 60, 12)
    total_relationship_count = st.sidebar.slider('Total Relationship Count', 1, 10, 3)
    months_inactive_12_mon = st.sidebar.slider('Months Inactive (12 months)', 0, 6, 2)
    contacts_count_12_mon = st.sidebar.slider('Contacts Count (12 months)', 0, 10, 3)
    credit_limit = st.sidebar.slider('Credit Limit', 0, 100000, 5000)
    total_revolving_bal = st.sidebar.slider('Total Revolving Balance', 0, 30000, 1000)
    total_amt_chng_q4_q1 = st.sidebar.slider('Total Amount Change (Q4-Q1)', 0.0, 2.0, 0.5)
    total_trans_amt = st.sidebar.slider('Total Transaction Amount', 0, 10000, 1000)
    total_trans_ct = st.sidebar.slider('Total Transaction Count', 0, 100, 20)
    total_ct_chng_q4_q1 = st.sidebar.slider('Total Count Change (Q4-Q1)', 0.0, 2.0, 0.5)
    avg_utilization_ratio = st.sidebar.slider('Average Utilization Ratio', 0.0, 1.0, 0.3)

    # Store the input data into a DataFrame
    customer_data = pd.DataFrame({
        'Customer_Age': [customer_age],
        'Gender': [gender],
        'Dependent_count': [dependent_count],
        'Education_Level': [education_level],
        'Marital_Status': [marital_status],
        'Income_Category': [income_category],
        'Card_Category': [card_category],
        'Months_on_book': [months_on_book],
        'Total_Relationship_Count': [total_relationship_count],
        'Months_Inactive_12_mon': [months_inactive_12_mon],
        'Contacts_Count_12_mon': [contacts_count_12_mon],
        'Credit_Limit': [credit_limit],
        'Total_Revolving_Bal': [total_revolving_bal],
        'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
        'Total_Trans_Amt': [total_trans_amt],
        'Total_Trans_Ct': [total_trans_ct],
        'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
        'Avg_Utilization_Ratio': [avg_utilization_ratio]
    })

    # Button to predict churn
    if st.sidebar.button('Predict Churn'):
        prediction, probability = predict_churn(customer_data)
        churn_status = 'Churn' if prediction[0] == 1 else 'Not Churn'
        st.success(f'The predicted churn status is: {churn_status} with probability: {probability:.2f}')

if __name__ == '__main__':
    main()
