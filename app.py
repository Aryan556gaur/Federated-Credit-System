import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from models.federated_model import create_model
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Federated Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)  # Old version
def load_resources():
    """Load model and preprocessing objects."""
    try:
        # Load preprocessing objects
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        target_encoder = joblib.load('models/target_encoder.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        
        # Create and load model
        model = create_model((len(feature_columns),), 3)
        if os.path.exists('models/federated_credit_model_weights.weights.h5'):
            model.load_weights('models/federated_credit_model_weights.weights.h5')
        else:
            st.error("Model weights not found. Please train the model first.")
            return None, None, None, None, None
        
        return model, scaler, label_encoders, target_encoder, feature_columns
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None, None

def predict_credit_score(input_data, model, scaler, label_encoders, target_encoder, feature_columns):
    """Make a credit score prediction."""
    input_df = pd.DataFrame([input_data])
    
    # Preprocess categorical columns
    categorical_columns = [col for col in feature_columns if col in label_encoders]
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
    
    # Scale numerical features
    numerical_columns = [col for col in feature_columns if col not in categorical_columns]
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    
    # Ensure correct feature order
    input_df = input_df[feature_columns]
    
    # Make prediction
    probabilities = model.predict(input_df.values)
    predicted_class = np.argmax(probabilities, axis=1)
    credit_score = target_encoder.inverse_transform(predicted_class)[0]
    
    return credit_score, probabilities[0]

def main():
    """Main application function."""
    st.title("🏦 Federated Credit Scoring System")
    st.markdown("""
    This system predicts credit scores using a federated learning model trained across multiple financial institutions 
    without sharing raw customer data.
    """)
    
    # Load resources
    model, scaler, label_encoders, target_encoder, feature_columns = load_resources()
    if model is None:
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Credit Score Prediction", "Model Information"])
    
    with tab1:
        st.header("Customer Credit Assessment")
        
        with st.form("credit_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Personal Information")
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                occupation = st.selectbox("Occupation", [
                    "Scientist", "Teacher", "Engineer", "Entrepreneur", 
                    "Developer", "Lawyer", "Media_Manager", "Doctor",
                    "Journalist", "Manager", "Accountant", "Musician",
                    "Mechanic", "Writer", "Architect"
                ])
                monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
                num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=2)
                num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, value=2)
                interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
                num_loans = st.number_input("Number of Loans", min_value=0, value=1)
                
            with col2:
                st.subheader("Financial History")
                delay_from_due_date = st.number_input("Average Delay From Due Date (days)", min_value=0, value=5)
                num_delayed_payments = st.number_input("Number of Delayed Payments", min_value=0, value=2)
                changed_credit_limit = st.selectbox("Changed Credit Limit Recently", ["Yes", "No"])
                num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=2)
                outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0, value=5000)
                credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.3)
                credit_history_age = st.number_input("Credit History Age (years)", min_value=0, value=5)
                payment_min_amount = st.selectbox("Typically Pays Minimum Amount", ["Yes", "No"])
                payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No"])
            
            submitted = st.form_submit_button("Predict Credit Score")
        
        if submitted:
            input_data = {
                "Age": age,
                "Occupation": occupation,
                "Monthly_Inhand_Salary": monthly_income,
                "Num_Bank_Accounts": num_bank_accounts,
                "Num_Credit_Card": num_credit_cards,
                "Interest_Rate": interest_rate,
                "Num_of_Loan": num_loans,
                "Delay_from_due_date": delay_from_due_date,
                "Num_of_Delayed_Payment": num_delayed_payments,
                "Changed_Credit_Limit": changed_credit_limit,
                "Num_Credit_Inquiries": num_credit_inquiries,
                "Outstanding_Debt": outstanding_debt,
                "Credit_Utilization_Ratio": credit_utilization_ratio,
                "Credit_History_Age": credit_history_age,
                "Payment_of_Min_Amount": payment_of_min_amount,
                "Payment_Behaviour": "High_spent_Medium_value_payments",
                "Monthly_Balance": monthly_income - outstanding_debt/12,
                "Amount_invested_monthly": 0,
                "Type_of_Loan": "Auto Loan",
                "Total_EMI_per_month": outstanding_debt/12 if num_loans > 0 else 0,
                "Annual_Income": monthly_income * 12
            }
            
            credit_score, probabilities = predict_credit_score(
                input_data, model, scaler, label_encoders, 
                target_encoder, feature_columns
            )
            
            # Display results
            st.subheader("Prediction Results")
            score_color = "green" if credit_score == "Good" else "orange" if credit_score == "Standard" else "red"
            st.markdown(f"""
            <div style="border: 2px solid {score_color}; border-radius: 5px; padding: 20px; text-align: center;">
                <h2 style="color: {score_color};">Predicted Credit Score: <strong>{credit_score}</strong></h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("Probability Breakdown:")
            prob_df = pd.DataFrame({
                "Credit Score": ["Good", "Standard", "Poor"],
                "Probability": probabilities
            })
            st.bar_chart(prob_df.set_index("Credit Score"))
            
            st.subheader("Interpretation")
            if credit_score == "Good":
                st.success("This customer has a high creditworthiness.")
            elif credit_score == "Standard":
                st.warning("This customer has moderate creditworthiness.")
            else:
                st.error("This customer has poor creditworthiness.")
    
    with tab2:
        st.header("Model Information")
        st.markdown("""
        ### Federated Learning Credit Scoring Model
        This model was trained using federated learning across multiple financial institutions.
        
        **Model Architecture:**
        - Input Layer: 23 features
        - Hidden Layers: 128, 64, and 32 neurons with ReLU activation
        - Dropout Layers: 30% and 20% dropout for regularization
        - Output Layer: 3 neurons with Softmax activation
        
        **Training Details:**
        - Number of Clients: 5
        - Training Rounds: 20
        - Learning Rate: 0.001
        - Batch Size: 32
        """)

if __name__ == "__main__":
    main()