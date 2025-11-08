import streamlit as st
import pandas as pd
from PIL import Image
import os

st.set_page_config(page_title="XGBoost Results", layout="wide")

st.title("🌲 XGBoost Model Results")

st.markdown("""
This page presents the results of the **XGBoost (Extreme Gradient Boosting)** model, a powerful ensemble learning method 
that builds multiple decision trees sequentially to correct errors from previous trees. XGBoost is known for its 
high performance in fraud detection tasks due to its ability to handle imbalanced data and capture complex patterns.
""")

# Load metrics
PLOTS_DIR = "plots"
metrics_path = os.path.join(PLOTS_DIR, "xgboost_metrics.csv")

if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    
    # Display metrics
    st.header("📊 Model Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        f1_score = metrics_df[metrics_df['Metric'] == 'F1-Score']['Value'].values[0]
        st.metric(
            label="F1-Score",
            value=f"{f1_score:.4f}",
            help="Harmonic mean of precision and recall"
        )
    
    with col2:
        pr_auc = metrics_df[metrics_df['Metric'] == 'PR-AUC']['Value'].values[0]
        st.metric(
            label="PR-AUC (Precision-Recall AUC)",
            value=f"{pr_auc:.4f}",
            help="Area under the precision-recall curve"
        )
    
    st.markdown("---")

    
    # Visualizations
    st.header("📊 Model Visualizations")
    
    # Confusion Matrix
    st.subheader("1️⃣ Confusion Matrix")
    st.markdown("""
    **How to read this plot:**
    - **True Positives (bottom right):** Correctly identified frauds
    - **True Negatives (top left):** Correctly identified non-fraud
    - **False Positives (top right):** Non-fraud flagged as fraud (false alarm)
    - **False Negatives (bottom left):** Missed frauds (dangerous)
    - **Goal:** High numbers on diagonal, low off-diagonal
    """)
    confusion_matrix_path = os.path.join(PLOTS_DIR, "xgboost_confusion_matrix.png")
    if os.path.exists(confusion_matrix_path):
        img = Image.open(confusion_matrix_path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Confusion matrix plot not found at {confusion_matrix_path}")

    st.markdown("---")
    
    # ROC Curve
    st.subheader("2️⃣ ROC Curve")
    st.markdown("""
    **How to read this plot:**
    - **Curve above diagonal:** Model performs better than random
    - **Area under curve (AUC):** Higher is better
    - **Goal:** Curve hugs top-left corner (high TPR, low FPR)
    """)
    roc_curve_path = os.path.join(PLOTS_DIR, "xgboost_roc_curve.png")
    if os.path.exists(roc_curve_path):
        img = Image.open(roc_curve_path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"ROC curve not found at {roc_curve_path}")

    st.markdown("---")
    
    # Precision-Recall Curve
    st.subheader("3️⃣ Precision-Recall Curve")
    st.markdown("""
    **How to read this plot:**
    - **Precision:** Accuracy of fraud alerts
    - **Recall:** % of fraud caught
    - **Goal:** Curve near top-right (high precision and recall)
    - **More informative than ROC for imbalanced data**
    """)
    pr_curve_path = os.path.join(PLOTS_DIR, "xgboost_pr_curve.png")
    if os.path.exists(pr_curve_path):
        img = Image.open(pr_curve_path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Precision-Recall curve not found at {pr_curve_path}")

    st.markdown("---")
    
    # Feature Importance
    st.subheader("4️⃣ Feature Importance")
    st.markdown("""
    **How to read this plot:**
    - **Higher bars:** More important features for fraud detection
    - **Shows which transaction characteristics drive model decisions**
    - **Common patterns:** Amount, transaction type, balance changes
    """)
    feature_importance_path = os.path.join(PLOTS_DIR, "xgboost_feature_importance.png")
    if os.path.exists(feature_importance_path):
        img = Image.open(feature_importance_path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Feature importance plot not found at {feature_importance_path}")
    
    st.markdown("---")
    
    # Model Insights and Conclusions
    st.header("💡 Key Insights & Conclusions")
    
    # Get metric values
    f1_score = metrics_df[metrics_df['Metric'] == 'F1-Score']['Value'].values[0] if 'F1-Score' in metrics_df['Metric'].values else 0
    pr_auc = metrics_df[metrics_df['Metric'] == 'PR-AUC']['Value'].values[0] if 'PR-AUC' in metrics_df['Metric'].values else 0
    
    st.markdown(f"""
### Model Performance Summary:
- **F1-Score: {f1_score:.4f}** - {'Strong' if f1_score > 0.7 else 'Moderate' if f1_score > 0.5 else 'Low'} balance between precision and recall
- **PR-AUC: {pr_auc:.4f}** - {'Excellent' if pr_auc > 0.85 else 'Good' if pr_auc > 0.7 else 'Moderate' if pr_auc > 0.5 else 'Poor'} performance on the precision-recall tradeoff for fraud detection

### Performance Analysis:

#### {'✅' if pr_auc > 0.7 else '🟡' if pr_auc > 0.5 else '❌'} XGBoost Strengths:
1. **Ensemble Learning**: Combines multiple decision trees to capture complex fraud patterns
2. **Gradient Boosting**: Each tree corrects errors from previous trees, improving overall accuracy
3. **Handles Imbalance**: Built-in mechanisms (`scale_pos_weight`) to address fraud/non-fraud class imbalance
4. **Feature Interactions**: Automatically learns relationships between features without manual engineering
5. **Regularization**: L1/L2 regularization prevents overfitting and improves generalization

#### 📊 Understanding the Metrics:
- **F1-Score ({f1_score:.4f})**: Harmonic mean of precision and recall
  - {'Strong performance - excellent balance between catching fraud and minimizing false alarms' if f1_score > 0.7 else 'Moderate performance - reasonable balance but room for improvement' if f1_score > 0.5 else 'Low performance - struggles to balance precision and recall'}
  - Precision: Accuracy when model flags fraud
  - Recall: Percentage of actual fraud cases caught
  
- **PR-AUC ({pr_auc:.4f})**: Measures the area under the precision-recall curve
  - For imbalanced datasets, this is the **most informative metric**
  - {'Excellent - model effectively distinguishes fraud from non-fraud' if pr_auc > 0.85 else 'Good - model captures meaningful fraud patterns' if pr_auc > 0.7 else 'Moderate - model shows learning but has limitations' if pr_auc > 0.5 else 'Poor - model struggles to detect fraud patterns'}
  - Baseline random guessing ≈ fraud rate in dataset (typically < 0.01)
  - {'This model is significantly better than random' if pr_auc > 0.5 else 'This model needs improvement'}

#### 🔍 How XGBoost Achieves These Results:
- **Sequential Tree Building**: Each tree focuses on correcting previous trees' mistakes
- **Gradient Descent**: Optimizes loss function to minimize prediction errors
- **Feature Importance**: Identifies which transaction characteristics matter most
- **Automatic Interaction Detection**: Learns complex patterns like "high amount + new merchant = high risk"
- **Class Weighting**: Adjusts for imbalanced fraud/non-fraud distribution

### Model Interpretation:
- **XGBoost** uses gradient boosting to build an ensemble of decision trees
- Each tree adds to the prediction, correcting errors from previous trees
- **Feature importance** reveals which transaction characteristics drive fraud detection
- **Tree depth and learning rate** control model complexity vs. generalization

### Conclusions:
{'🟢' if pr_auc > 0.7 else '🟡' if pr_auc > 0.5 else '🔴'} **XGBoost {'is suitable' if pr_auc > 0.7 else 'shows promise but needs improvement' if pr_auc > 0.5 else 'requires significant improvement'} for fraud detection:**
- {'PR-AUC of ' + f'{pr_auc:.4f}' + ' indicates ' + ('strong' if pr_auc > 0.7 else 'moderate' if pr_auc > 0.5 else 'limited') + ' fraud detection capability'}
- {'F1-Score of ' + f'{f1_score:.4f}' + ' shows ' + ('excellent' if f1_score > 0.7 else 'reasonable' if f1_score > 0.5 else 'limited') + ' balance between precision and recall'}
- {'Ensemble approach effectively captures complex fraud patterns' if pr_auc > 0.7 else 'Gradient boosting shows improvement over simpler models' if pr_auc > 0.5 else 'Consider hyperparameter tuning or different approaches'}
- {'Strong candidate for production deployment' if pr_auc > 0.85 and f1_score > 0.7 else 'May benefit from ensemble combination with other models' if pr_auc > 0.5 else 'Requires further optimization before production use'}

"""
)

else:
    st.error(f"Metrics file not found at {metrics_path}. Please run the XGBoost training notebook first.")
    st.info("Expected file: `plots/xgboost_metrics.csv`")

st.divider()

# Footer
st.markdown("---")
st.markdown("*Model trained using XGBoost on the financial fraud detection dataset*")