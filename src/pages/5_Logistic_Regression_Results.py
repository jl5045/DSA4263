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
st.markdown("""
**How to read this plot:**
- **True Positives (bottom right):** Correctly identified frauds
- **True Negatives (top left):** Correctly identified non-fraud
- **False Positives (top right):** Non-fraud flagged as fraud (false alarm)
- **False Negatives (bottom left):** Missed frauds (dangerous)
- **Goal:** High numbers on diagonal, low off-diagonal
""")
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
    st.markdown("""
    **How to read this plot:**
    - **Curve above diagonal:** Model performs better than random
    - **Area under curve (AUC):** Higher is better
    - **Goal:** Curve hugs top-left corner (high TPR, low FPR)
    """)
    roc_curve_path = os.path.join(plots_dir, "logistic_regression_roc_curve.png")
    if os.path.exists(roc_curve_path):
        img = Image.open(roc_curve_path)
        st.image(img, use_container_width=True)
    else:
        st.warning("⚠️ ROC curve not found.")

with col2:
    st.subheader("Precision-Recall Curve")
    st.markdown("""
    **How to read this plot:**
    - **Precision:** Accuracy of fraud alerts
    - **Recall:** % of fraud caught
    - **Goal:** Curve near top-right (high precision and recall)
    - **More informative than ROC for imbalanced data**
    """)
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
**How to read this plot:**
- **Green bars:** Features that increase fraud probability
- **Red bars:** Features that decrease fraud probability
- **Longer bars:** More influential features
- **Goal:** Identify which features most strongly affect model predictions
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
- **PR-AUC: 0.0185** - Very poor performance on the precision-recall tradeoff for fraud detection

### Critical Analysis:

#### ❌ Why the Model Performs Poorly:
1. **Severe Class Imbalance**: The dataset is heavily skewed toward non-fraud cases
2. **Low Precision**: The model produces many false positives (non-fraud predicted as fraud)
3. **Low Recall**: The model misses many actual fraud cases (false negatives)
4. **Extremely Low PR-AUC**: A score of 0.0185 indicates the model barely outperforms random guessing on the fraud class

#### 📊 Understanding the Metrics:
- **F1-Score (0.0107)**: Harmonic mean of precision and recall - extremely low suggests the model cannot reliably identify fraud
- **PR-AUC (0.0185)**: Measures the area under the precision-recall curve
  - For imbalanced datasets, this is the **most informative metric**
  - A score near 0 means the model has almost no ability to distinguish fraud from non-fraud
  - Random guessing would achieve ~0.001 (baseline fraud rate), so 0.0185 is only marginally better

#### 🔍 Why Both Metrics Are So Low:
- The model's **linear decision boundary** cannot capture complex fraud patterns
- **Balanced class weights** are insufficient to overcome the extreme imbalance
- Simple features from raw data lack the discriminative power needed
- The model likely predicts "non-fraud" for most cases to minimize error on the majority class

### Model Interpretation:
- **Logistic Regression** provides interpretable coefficients for each feature
- The model uses **balanced class weights** to handle imbalance, but it's insufficient
- **Simple linear decision boundary** cannot capture complex fraud patterns
- Serves as a **sanity check baseline** - any viable model must significantly outperform these scores

### Conclusions:
🔴 **Logistic Regression is NOT suitable for this fraud detection task:**
- PR-AUC of 0.0185 indicates near-zero fraud detection capability
- F1-Score of 0.0107 means almost no true fraud is correctly identified
- Linear assumptions fundamentally incompatible with fraud patterns

"""
)
