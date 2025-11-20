import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Neural Network Results", layout="wide")

st.title("🧠 Neural Network Model Results")

st.markdown("""
This page displays the results of the Neural Network model trained on the fraud detection dataset.
The model uses deep learning architecture to capture complex patterns in the transaction data.
""")

# Define paths
plots_dir = "./plots"
metrics_path = os.path.join(plots_dir, "neural_network_metrics.csv")

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
    st.warning("⚠️ Metrics file not found. Please run the `4_Neural_Network.ipynb` notebook first.")

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
confusion_matrix_path = os.path.join(plots_dir, "neural_network_confusion_matrix.png")
if os.path.exists(confusion_matrix_path):
    img = Image.open(confusion_matrix_path)
    st.image(img, use_container_width=True)
else:
    st.warning("⚠️ Confusion matrix plot not found. Please run the notebook to generate it.")

st.divider()

# Section 3: Precision-Recall Curve
st.header("📈 Performance Curve")
st.subheader("Precision-Recall Curve")
st.markdown("""
**How to read this plot:**
- **Precision:** Accuracy of fraud alerts
- **Recall:** % of fraud caught
- **Goal:** Curve near top-right (high precision and recall)
- **More informative than ROC for imbalanced data**
""")
pr_curve_path = os.path.join(plots_dir, "neural_network_pr_curve.png")
if os.path.exists(pr_curve_path):
    img = Image.open(pr_curve_path)
    st.image(img, use_container_width=True)
else:
    st.warning("⚠️ PR curve not found.")

st.divider()

# Section 4: Feature Importance
st.header("🔢 Feature Importance")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Standard Feature Importance")
    st.markdown("""
    **How to read this plot:**
    - **Higher bars:** More important features for fraud detection
    - **Shows which transaction characteristics drive model decisions**
    - **Note:** For deep networks, these raw first-layer weights are only a proxy for importance
    """)
    feature_importance_path = os.path.join(plots_dir, "neural_network_feature_importance.png")
    if os.path.exists(feature_importance_path):
        img = Image.open(feature_importance_path)
        st.image(img, use_container_width=True)
    else:
        st.warning("⚠️ Feature importance plot not found. Please run the notebook to generate it.")

with col2:
    st.subheader("SHAP Feature Importance")
    st.markdown("""
    **How to read this plot:**
    - **Higher bars:** Features with greater impact on model predictions
    - **SHAP values:** Show average effect of each feature on output
    - **Goal:** Identify which features most strongly affect model predictions
    """)
    feature_importance_shap_path = os.path.join(plots_dir, "neural_network_feature_importance_SHAP.png")
    if os.path.exists(feature_importance_shap_path):
        img = Image.open(feature_importance_shap_path)
        st.image(img, use_container_width=True)
    else:
        st.warning("⚠️ SHAP feature importance plot not found. Please run the notebook to generate it.")

st.divider()

# Section 5: Model Insights and Conclusions
st.header("💡 Key Insights & Conclusions")

st.markdown("""
### Model Performance Summary:
- **F1-Score: 0.5143** - Moderate balance between precision and recall
- **PR-AUC: 0.5158** - Moderate discrimination ability on the precision-recall curve

### Performance Analysis:

#### ✅ Neural Network Strengths:
1. **Significant Improvement Over Baseline**: PR-AUC of 0.5158 vs. Logistic Regression's 0.0185 shows massive improvement
2. **Non-linear Pattern Recognition**: Captures complex relationships between features that linear models cannot
3. **Deep Learning Architecture**: Multiple layers enable hierarchical feature learning
4. **Moderate F1-Score**: Shows reasonable ability to balance precision and recall

#### ⚠️ Performance Limitations:
1. **Still Moderate Performance**: PR-AUC of 0.5158 indicates room for improvement
2. **Precision-Recall Trade-off**: The model struggles to achieve both high precision AND high recall simultaneously
3. **Class Imbalance Challenges**: Even with deep learning, the severe fraud/non-fraud imbalance affects performance

#### 📊 Understanding the Metrics:
- **F1-Score (0.5143)**: Reasonable balance between precision and recall, better than baseline but not optimal
- **PR-AUC (0.5158)**: For this imbalanced dataset, a score around 0.5 indicates the model is learning meaningful patterns but has substantial room for improvement
  - Baseline fraud rate would achieve ~0.001
  - This model is 500+ times better than random, but still has limitations

#### 🔍 Key Insights:
- The neural network learns **non-linear decision boundaries** more effectively than linear models
- **SHAP feature importance** provides more reliable insights than raw first-layer weights
- Model captures transaction patterns better, but may benefit from ensemble approaches or further optimization

### Model Interpretation:
- **Feature Importance**: Shows which patterns the deep network learned
- **SHAP Values**: Provides interpretable, reliable feature importance (preferable to raw weights)
- **Confusion Matrix**: Reveals the practical precision-recall trade-off
- **PR Curve**: Shows how performance varies at different decision thresholds

### Conclusions:
🟡 **Neural Network Shows Significant Improvement:**
- PR-AUC of 0.5158 demonstrates substantial improvement over Logistic Regression baseline
- Deep learning successfully captures non-linear fraud patterns
- Moderate performance suggests potential for ensemble methods or hyperparameter tuning
- SHAP values provide interpretable insights into model decisions
""")

