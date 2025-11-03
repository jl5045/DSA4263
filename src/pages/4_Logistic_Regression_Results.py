import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Logistic Regression Results", layout="wide")

st.title("🔍 Logistic Regression Model Results")

st.markdown("""
This page displays the results of the Logistic Regression model trained on the raw fraud detection dataset.
This model will be our baseline model measure on F1 score and PR-AUC.
""")

# Define paths
plots_dir = "./plots"
metrics_path = os.path.join(plots_dir, "logistic_regression_metrics.csv")

# Load metrics
if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    
    # Section 1: Overall Metrics
    st.header("📊 Model Performance Metrics")
    
    # Display metrics in columns
    cols = st.columns(len(metrics_df))
    for idx, (col, row) in enumerate(zip(cols, metrics_df.itertuples())):
        with col:
            st.metric(label=row.Metric, value=f"{row.Value:.4f}")
    
    st.divider()
else:
    st.warning("⚠️ Metrics file not found. Please run the `3_Logistic_Regression.ipynb` notebook first.")

# Section 2: Confusion Matrix
st.header("🎯 Confusion Matrix")
confusion_matrix_path = os.path.join(plots_dir, "logistic_regression_confusion_matrix.png")
if os.path.exists(confusion_matrix_path):
    img = Image.open(confusion_matrix_path)
    st.image(img, use_container_width=True)
else:
    st.warning("⚠️ Confusion matrix plot not found. Please run the notebook to generate it.")

st.divider()

# Section 3: ROC Curve and PR Curve
st.header("📈 Performance Curves")
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curve")
    roc_curve_path = os.path.join(plots_dir, "logistic_regression_roc_curve.png")
    if os.path.exists(roc_curve_path):
        img = Image.open(roc_curve_path)
        st.image(img, use_container_width=True)
    else:
        st.warning("⚠️ ROC curve not found.")

with col2:
    st.subheader("Precision-Recall Curve")
    pr_curve_path = os.path.join(plots_dir, "logistic_regression_pr_curve.png")
    if os.path.exists(pr_curve_path):
        img = Image.open(pr_curve_path)
        st.image(img, use_container_width=True)
    else:
        st.warning("⚠️ PR curve not found.")

st.divider()

# Section 4: Feature Importance
st.header("🔢 Feature Importance")
st.markdown("""
The coefficients show how each feature contributes to the model's prediction. 
- **Green bars**: Positive coefficients (increase fraud probability)
- **Red bars**: Negative coefficients (decrease fraud probability)
""")

feature_importance_path = os.path.join(plots_dir, "logistic_regression_feature_importance.png")
if os.path.exists(feature_importance_path):
    img = Image.open(feature_importance_path)
    st.image(img, use_container_width=True)
else:
    st.warning("⚠️ Feature importance plot not found. Please run the notebook to generate it.")

st.divider()

# Section 5: Model Insights and Conclusions
st.header("💡 Key Insights & Conclusions")

st.markdown("""
### Model Performance Summary:
- **F1-Score: 0.0107** - Extremely low F1-score indicates poor balance between precision and recall
- **ROC-AUC: 0.8262** - Moderate discriminative ability, but misleading for heavily imbalanced datasets

### Critical Analysis:

#### Why the Model Performs Poorly:
1. **Severe Class Imbalance**: The dataset is heavily skewed toward non-fraud cases
2. **Low Precision**: The model produces many false positives (non-fraud predicted as fraud)
3. **Low Recall**: The model misses many actual fraud cases (false negatives)
4. **Misleading ROC-AUC**: While ROC-AUC is 0.83, it doesn't reflect real-world performance on the minority class

#### F1-Score vs ROC-AUC Discrepancy:
- **ROC-AUC** measures overall ranking ability but treats both classes equally
- **F1-Score** directly measures performance on the positive (fraud) class
- The large gap (0.83 vs 0.01) reveals the model struggles specifically with fraud detection despite decent ranking

### Model Interpretation:
- **Logistic Regression** provides interpretable coefficients for each feature
- The model uses **balanced class weights** to handle imbalance, but it's insufficient
- **Simple linear decision boundary** cannot capture complex fraud patterns

### Conclusions:
**Logistic Regression is NOT suitable as a production model** for this fraud detection task due to:
- Unacceptably low F1-score (1%)
- Poor real-world fraud detection performance
- Linear assumptions that don't match complex fraud behaviors

"""
)
