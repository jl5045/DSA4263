import streamlit as st
import os
from pathlib import Path

st.set_page_config(
    page_title="Explainable AI - SHAP Analysis",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Explainable AI: SHAP Analysis for XGBoost")

st.markdown("""
Understanding **why** a model makes certain predictions is crucial for fraud detection systems. 
This page uses **SHAP (SHapley Additive exPlanations)** to explain how our XGBoost model identifies fraudulent transactions.

SHAP values tell us the contribution of each feature to the model's prediction, providing transparency and interpretability.
""")

plots_dir = "./plots"

# Check if SHAP plots exist
shap_plots = {
    "beeswarm": os.path.join(plots_dir, "xgb_shap_beeswarm.png"),
    "bar": os.path.join(plots_dir, "xgb_shap_bar.png"),
    "waterfall": os.path.join(plots_dir, "xgb_shap_waterfall_fraud_example.png"),
    "force": os.path.join(plots_dir, "xgb_shap_force_fraud_example.html")
}

# SHAP Feature Importance (Bar Plot)
st.header("📊 Global Feature Importance")
st.markdown("""
This plot shows which features are most important across **all predictions** in the dataset.
The features are ranked by their average impact on the model's output.
""")

if os.path.exists(shap_plots["bar"]):
    st.image(shap_plots["bar"], use_container_width=True)
    
    with st.expander("ℹ️ How to Read This Plot"):
        st.markdown("""
        **What it shows:**
        - Features are ranked from most to least important (top to bottom)
        - The x-axis shows the mean absolute SHAP value (average impact on predictions)
        - Higher values mean the feature has a stronger influence on whether a transaction is classified as fraud
        
        **Key insights:**
        - The top features are the most critical for fraud detection
        - These features should be monitored closely in production
        - If these features are missing or corrupted, model performance will degrade significantly
        """)
else:
    st.warning("SHAP bar plot not found. Please ensure `xgb_shap_bar.png` exists in the plots directory.")

st.markdown("---")

# SHAP Beeswarm Plot
st.header("🎯 Feature Impact Distribution")
st.markdown("""
This plot shows **how** each feature impacts predictions across different transactions.
Each point represents a single transaction, colored by the feature's value.
""")

if os.path.exists(shap_plots["beeswarm"]):
    st.image(shap_plots["beeswarm"], use_container_width=True)
    
    with st.expander("ℹ️ How to Read This Plot"):
        st.markdown("""
        **What it shows:**
        - Features are ranked by importance (top to bottom)
        - Each dot is a transaction
        - **X-axis (SHAP value):** How much the feature pushes the prediction toward fraud (right) or non-fraud (left)
        - **Color:** Red = high feature value, Blue = low feature value
        
        **Key insights:**
        - **Red dots on the right:** High feature values → higher fraud probability
        - **Blue dots on the left:** Low feature values → lower fraud probability
        - **Spread:** Wide spread means the feature has varying impact across transactions
        
        **Example:**
        If "transaction amount" has many red dots on the right, it means high amounts strongly indicate fraud.
        """)
else:
    st.warning("SHAP beeswarm plot not found. Please ensure `xgb_shap_beeswarm.png` exists in the plots directory.")

st.markdown("---")

# SHAP Waterfall Plot (Individual Prediction)
st.header("💧 Individual Prediction Explanation (Waterfall)")
st.markdown("""
This plot explains **a single fraud prediction** by showing how each feature contributed to moving the prediction 
from the baseline (average prediction) to the final fraud probability.
""")

if os.path.exists(shap_plots["waterfall"]):
    st.image(shap_plots["waterfall"], use_container_width=True)
    
    with st.expander("ℹ️ How to Read This Plot"):
        st.markdown("""
        **What it shows:**
        - **Bottom (E[f(x)]):** The baseline/average prediction across all transactions
        - **Top (f(x)):** The final prediction for this specific transaction
        - **Bars:** Each feature's contribution (red = pushes toward fraud, blue = pushes toward non-fraud)
        - Features are ordered by impact magnitude
        
        **How to interpret:**
        1. Start at the baseline (bottom)
        2. Each bar shows how a feature adjusts the prediction
        3. Red bars push the prediction higher (more likely fraud)
        4. Blue bars push the prediction lower (less likely fraud)
        5. Follow the bars upward to reach the final prediction
        
        **Example:**
        - If "transaction type = TRANSFER" adds +0.3 (red), it's a strong fraud indicator
        - If "account balance" adds -0.1 (blue), it slightly reduces fraud probability
        - The sum of all contributions gives the final fraud score
        """)
else:
    st.warning("SHAP waterfall plot not found. Please ensure `xgb_shap_waterfall_fraud_example.png` exists in the plots directory.")

st.markdown("---")

# SHAP Force Plot (Individual Prediction - Interactive)
st.header("⚡ Force Plot: Interactive Prediction Breakdown")
st.markdown("""
This interactive visualization shows the same fraud prediction but in a different format.
Features pushing toward fraud are shown in red, features pushing against fraud are shown in blue.
""")

if os.path.exists(shap_plots["force"]):
    with open(shap_plots["force"], 'r', encoding='utf-8') as f:
        force_plot_html = f.read()
    st.components.v1.html(force_plot_html, height=300, scrolling=True)
    
    with st.expander("ℹ️ How to Read This Plot"):
        st.markdown("""
        **What it shows:**
        - **Base value:** Starting point (average prediction)
        - **Red features:** Push the prediction toward fraud (positive SHAP values)
        - **Blue features:** Push the prediction toward non-fraud (negative SHAP values)
        - **Output value:** Final fraud probability prediction
        
        **How to interpret:**
        - Wider bars = stronger feature impact
        - Hover over features to see exact contribution values
        - The base value + all contributions = final prediction
        
        **Why this matters:**
        - Provides transparency: You can see exactly why the model flagged this transaction
        - Enables audit trails: Document why specific transactions were blocked
        - Builds trust: Fraud analysts can validate the model's reasoning
        """)
else:
    st.warning("SHAP force plot not found. Please ensure `xgb_shap_force_fraud_example.html` exists in the plots directory.")

st.markdown("---")

st.markdown("""
### What SHAP Analysis Reveals About Our Model

Based on the visualizations above, key insights about the XGBoost fraud detection model:

**Top Fraud Indicators:**
- **Transaction Type (TRANSFER/CASH_OUT):** Consistently appears as the strongest predictor. High SHAP values for these transaction types indicate they are primary fraud signals.
- **Transaction Amount Patterns:** Large amounts combined with specific transaction types create strong fraud signals. The model learns that certain amount ranges are more suspicious.
- **Account Balance Anomalies:** Unusual relationships between transaction amount and account balances (both origin and destination) flag potential fraud.
- **Network Features:** Behavioral patterns like transaction frequency, unique destinations, and account relationships help identify coordinated fraud rings.

**Model Behavior:**
- The **beeswarm plot** shows that the model doesn't rely on a single feature but combines multiple signals
- The **waterfall/force plots** demonstrate that fraud predictions result from accumulating evidence across many features
- No single feature alone determines fraud - it's the combination that matters

**Practical Implications:**
- Fraudsters cannot easily bypass detection by manipulating just one feature
- The model captures both individual transaction characteristics and network-level patterns
- Feature importance remains relatively stable, suggesting robust fraud detection rules

This multi-faceted approach makes the model resilient against adversarial attacks and fraud evolution.
""")

