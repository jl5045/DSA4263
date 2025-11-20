import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import os
from io import StringIO

st.set_page_config(
    page_title="Live Fraud Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Live Fraud Prediction with XGBoost")

st.markdown("""
This page allows you to upload transaction data or use sample data to get fraud predictions using the trained XGBoost model.
The model will provide a fraud probability score (0-1) for each transaction where higher scores indicate higher likelihood of fraud.

XGBoost was chosen due to it's quick inference time and strong performance on tabular data, making it suitable for real-time fraud detection scenarios.
""")

# Load XGBoost model
@st.cache_resource
def load_xgb_model():
    model_path = "./models/XGB/best_xgb_1to5_all.json"
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    else:
        return None

model = load_xgb_model()

if model is None:
    st.error("❌ XGBoost model not found. Please train the model first.")
    st.info("Expected file: `models/XGB/best_xgb_1to5_all.json`")
else:
    st.success("✅ XGBoost model loaded successfully!")
    
    st.markdown("---")
    
    # Create sample dataset
    def create_sample_dataset():
        """Create a sample dataset for demonstration"""
        sample_data = {
            'step': [100, 200, 150, 300, 250],
            'type': [4, 1, 3, 2, 4],  # 0=CASH_IN, 1=CASH_OUT, 2=DEBIT, 3=PAYMENT, 4=TRANSFER
            'hourOfDay': [10, 14, 9, 18, 22],
            'day': [15, 20, 5, 25, 10],
            'amountLog': [7.5, 8.2, 6.8, 9.1, 7.0],
            'dayOfWeek': [2, 4, 1, 5, 3],
            'amount': [1000, 2500, 800, 5000, 1200],
            'oldbalanceOrg': [5000, 3000, 10000, 2000, 8000],
            'oldbalanceDest': [3000, 5000, 2000, 7000, 4000],
            'amount_to_oldbalanceOrg': [0.2, 0.83, 0.08, 2.5, 0.15],
            'meanSent': [600, 1200, 500, 800, 700],
            'totalSent': [6000, 12000, 5000, 8000, 7000],
            'stdSent': [300, 500, 200, 400, 350],
            'numSent': [10, 10, 10, 10, 10],
            'totalReceived': [4000, 3000, 5000, 2000, 6000],
            'numReceived': [8, 6, 10, 4, 12],
            'stdReceived': [400, 350, 250, 300, 450],
            'meanReceived': [500, 500, 500, 500, 500],
            'maxAmountReceived': [2000, 1500, 1800, 1200, 2200],
            'stdAmountReceived': [300, 250, 200, 150, 350],
            'std_to_mean_ratio': [0.5, 0.4, 0.3, 0.35, 0.6],
            'avgAmountToDest': [800, 900, 700, 1000, 850],
            'pctForwarded24h': [0.1, 0.2, 0.05, 0.15, 0.25],
            'pairFrequency': [5, 3, 8, 2, 7],
            'pctUniqueDest': [0.6, 0.5, 0.7, 0.4, 0.65],
            'pctUniqueOrig': [0.4, 0.3, 0.5, 0.2, 0.45],
            'transaction_sequence': [1, 2, 1, 3, 2],
            'sequence_frequency': [10, 5, 12, 3, 8],
            'transactionRecency': [5, 10, 2, 20, 8],
            'typeHighValueFlag': [0, 1, 0, 1, 0],
            'is_early_transaction': [0, 0, 1, 0, 1],
            'sequence_count': [1, 2, 1, 3, 2],
            'is_transfer_cashout': [1, 0, 0, 0, 1],
            'is_cashin_transfer': [0, 0, 0, 0, 0],
            'is_cashout_transfer': [0, 1, 0, 0, 0],
            'is_cashin_transfer_cashout': [0, 0, 0, 0, 0],
            'is_transfer_transfer': [1, 0, 0, 0, 1],
            'is_first_transfer': [1, 0, 0, 1, 0],
            'is_cashin_cashout': [0, 0, 0, 0, 0],
            'sender_btwn': [0.05, 0.08, 0.03, 0.10, 0.06],
            'receiver_btwn': [0.04, 0.06, 0.02, 0.09, 0.05],
            'btwn_diff': [0.01, 0.02, 0.01, 0.01, 0.01],
            'sender_outdeg_amt': [50000, 30000, 80000, 20000, 60000],
            'sender_indeg_amt': [5000, 8000, 3000, 12000, 7000],
            'receiver_outdeg_amt': [10000, 15000, 5000, 20000, 8000],
            'receiver_indeg_amt': [45000, 25000, 70000, 15000, 55000],
            'sender_outdeg_cnt': [50, 30, 80, 20, 60],
            'sender_indeg_cnt': [10, 15, 8, 20, 12],
            'receiver_outdeg_cnt': [15, 20, 10, 25, 18],
            'receiver_indeg_cnt': [40, 25, 75, 18, 50],
            'outdeg_amt_diff': [40000, 15000, 75000, 0, 52000],
            'indeg_amt_diff': [-40000, -17000, -65000, -8000, -48000],
            'outdeg_cnt_diff': [35, 10, 70, -5, 42],
            'indeg_cnt_diff': [-30, -10, -67, 2, -38],
        }
        return pd.DataFrame(sample_data)
    
    # Data input method
    st.header("📊 Data Input")
    
    data_input_method = st.radio("Choose data input method:", ["📤 Upload CSV", "📋 Use Sample Data"])
    
    data = None
    
    if data_input_method == "📤 Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type="csv")
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(data)} transactions")
                st.dataframe(data.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("Using sample dataset with 5 sample transactions")
        data = create_sample_dataset()
        st.dataframe(data, use_container_width=True)
    
    st.markdown("---")
    
    # Make predictions
    if data is not None and st.button("🔮 Predict Fraud for All Transactions", key="predict_button"):
        try:
            # Create DMatrix for prediction
            dmatrix = xgb.DMatrix(data)
            
            # Make predictions
            fraud_probabilities = model.predict(dmatrix)
            
            # Add predictions to dataframe
            results_df = data.copy()
            results_df['fraud_probability'] = fraud_probabilities
            results_df['fraud_probability_pct'] = (fraud_probabilities * 100).round(2)
            
            # Classify risk level
            def classify_risk(prob):
                if prob > 0.7:
                    return "🔴 HIGH"
                elif prob > 0.4:
                    return "🟡 MEDIUM"
                else:
                    return "🟢 LOW"
            
            results_df['risk_level'] = results_df['fraud_probability'].apply(classify_risk)
            
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            high_risk = (fraud_probabilities > 0.7).sum()
            medium_risk = ((fraud_probabilities > 0.4) & (fraud_probabilities <= 0.7)).sum()
            low_risk = (fraud_probabilities <= 0.4).sum()
            avg_fraud_prob = fraud_probabilities.mean()
            
            with col1:
                st.metric("🔴 High Risk", high_risk)
            with col2:
                st.metric("🟡 Medium Risk", medium_risk)
            with col3:
                st.metric("🟢 Low Risk", low_risk)
            with col4:
                st.metric("Average Fraud Probability", f"{avg_fraud_prob:.2%}")
            
            st.markdown("---")
            
            # Display results table
            st.subheader("� Transaction Predictions")
            display_cols = ['amount', 'fraud_probability_pct', 'risk_level', 'step', 'day', 'type']
            available_cols = [col for col in display_cols if col in results_df.columns]
            st.dataframe(
                results_df[available_cols].sort_values('fraud_probability_pct', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="� Download Results as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Risk breakdown
            st.subheader("📊 Risk Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **High Risk Transactions ({high_risk})**
                - Fraud Probability: > 70%
                - Action: ❌ Block immediately
                """)
            
            with col2:
                st.warning(f"""
                **Medium Risk Transactions ({medium_risk})**
                - Fraud Probability: 40-70%
                - Action: ⏸️ Send to manual review
                """)
            
            if low_risk > 0:
                st.success(f"""
                **Low Risk Transactions ({low_risk})**
                - Fraud Probability: < 40%
                - Action: ✓ Approve automatically
                """)
            
        except Exception as e:
            st.error(f"❌ Error making predictions: {str(e)}")
            st.info("Make sure all required features are present in your data.")
    
    st.markdown("---")
    
    # Information about required features
    st.subheader("ℹ️ Required Features")
    st.info("""
    The XGBoost model requires the following features:
    - **Time Features**: step, hourOfDay, day, dayOfWeek, amountLog
    - **Account Features**: amount, oldbalanceOrg, oldbalanceDest, amount_to_oldbalanceOrg
    - **Historical Features**: meanSent, totalSent, stdSent, numSent, totalReceived, numReceived, etc.
    - **Network Features**: sender_btwn, receiver_btwn, sender_outdeg_amt, etc.
    - **Behavioral Features**: typeHighValueFlag, is_early_transaction, pairFrequency, etc.
    
    Upload a CSV file with these columns, or use the sample data to test the predictions.
    """)
