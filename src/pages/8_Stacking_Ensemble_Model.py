import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Stacking Ensemble Model Results", layout="wide")

st.title("🔗 Stacking Ensemble Model Results")

st.markdown("""
This page displays the results of the Stacking Ensemble model, which combines multiple base models 
to improve fraud detection performance. The ensemble leverages the strengths of different algorithms 
to achieve better overall results than individual models.
""")

# Define paths
plots_dir = "./plots"
metrics_path = os.path.join(plots_dir, "stacking_ensemble_metrics.csv")

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
    st.warning("⚠️ Metrics file not found. Please run the ensemble model training notebook first.")

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
confusion_matrix_path = os.path.join(plots_dir, "stacking_ensemble_confusion_matrix.png")
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
    roc_curve_path = os.path.join(plots_dir, "stacking_ensemble_roc_curve.png")
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
    pr_curve_path = os.path.join(plots_dir, "stacking_ensemble_pr_curve.png")
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
- **Higher bars:** More important features for fraud detection
- **Shows which transaction characteristics drive model decisions**
- **Goal:** Identify which features most strongly affect model predictions
""")
feature_importance_path = os.path.join(plots_dir, "stacking_ensemble_feature_importance.png")
if os.path.exists(feature_importance_path):
    img = Image.open(feature_importance_path)
    st.image(img, use_container_width=True)
else:
    st.warning("⚠️ Feature importance plot not found. Please run the notebook to generate it.")

st.divider()

# Section 8: Model Insights and Conclusions
st.header("💡 Key Insights & Conclusions")

st.markdown("""
### Model Performance Summary:
- **F1-Score: 0.8211** - Very good balance between precision and recall
- **PR-AUC: 0.9026** - Strong discrimination ability on the precision-recall curve

### Performance Analysis:

#### ✅ Ensemble Strengths:
1. **Exceptional Performance**: PR-AUC of 0.9026 significantly outperforms individual models
2. **Excellent Precision-Recall Balance**: F1-Score of 0.8211 indicates the model successfully catches fraud while minimizing false alarms
3. **Combines Complementary Strengths**: Stacking leverages different algorithms to capture diverse fraud patterns
4. **Strong Discrimination**: High PR-AUC demonstrates excellent ability to rank fraud cases correctly

#### 📊 Performance Comparison:
1. **Logistic Regression Baseline**: PR-AUC = 0.0185 (poor linear model)
2. **Neural Network**: PR-AUC = 0.5158 (moderate non-linear model)
3. **Stacking Ensemble**: PR-AUC = 0.9026 (excellent combined approach)
   - **1.75x better than Neural Network alone**
   - **488x better than Logistic Regression baseline**

#### 🎯 Understanding the Metrics:
- **F1-Score (0.8211)**: Harmonic mean showing excellent precision-recall balance
  - Significantly higher than individual models
  - Indicates the model effectively identifies fraud without excessive false positives
- **PR-AUC (0.9026)**: For imbalanced datasets, this is the gold standard metric
  - Score near 0.9 indicates excellent discrimination between fraud and non-fraud
  - Model reliably identifies fraud cases across different decision thresholds

#### 🔍 Why Stacking Works Better:
- **Base Models Capture Different Patterns**: Each algorithm learns different fraud indicators
- **Meta-Learner Optimizes Combination**: Learns optimal weights to combine predictions
- **Reduces Individual Model Weaknesses**: Compensates for each model's limitations
- **Improves Generalization**: Ensemble reduces overfitting compared to individual models

### Model Interpretation:
- **Feature Importance**: Reflects the combined influence of features across all base models
- **Confusion Matrix**: Shows practical fraud detection capability at optimal threshold
- **ROC and PR Curves**: Demonstrate superior performance across all operating points
- **Ensemble Approach**: Meta-learner effectively balances precision and recall

### Conclusions:
🟢 **Stacking Ensemble is the Recommended Production Model:**
- PR-AUC of 0.9026 demonstrates exceptional fraud detection capability
- F1-Score of 0.8211 shows excellent balance between catching fraud and minimizing false alarms
- Significant improvement over all individual models
- Ensemble approach successfully combines the strengths of multiple algorithms for robust fraud detection
""")
