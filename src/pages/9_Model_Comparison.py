import streamlit as st
import pandas as pd
import os


st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Comparison")

plots_dir = "./plots"

# Define metric files for each model
metric_files = {
    "Logistic Regression": os.path.join(plots_dir, "logistic_regression_metrics.csv"),
    "Neural Network": os.path.join(plots_dir, "neural_network_metrics.csv"),
    "XGBoost": os.path.join(plots_dir, "xgboost_metrics.csv"),
    "Stacking Ensemble": os.path.join(plots_dir, "stacking_ensemble_metrics.csv"),
}

# Collect metrics
results = []
for model_name, file_path in metric_files.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        f1 = df[df['Metric'] == 'F1-Score']['Value'].values[0] if 'F1-Score' in df['Metric'].values else None
        pr_auc = df[df['Metric'] == 'PR-AUC']['Value'].values[0] if 'PR-AUC' in df['Metric'].values else None
        results.append({
            "Model": model_name,
            "F1-Score": f1,
            "PR-AUC": pr_auc
        })
    else:
        results.append({
            "Model": model_name,
            "F1-Score": None,
            "PR-AUC": None
        })

# Display comparison table
st.header("🔎 Model Comparison Table")
comparison_df = pd.DataFrame(results)
st.dataframe(comparison_df, use_container_width=True)

# Get XGBoost and Ensemble metrics
xgb_row = comparison_df[comparison_df['Model'] == 'XGBoost'].iloc[0]
xgb_f1 = xgb_row['F1-Score']
xgb_pr_auc = xgb_row['PR-AUC']

ensemble_row = comparison_df[comparison_df['Model'] == 'Stacking Ensemble'].iloc[0]
ensemble_f1 = ensemble_row['F1-Score']
ensemble_pr_auc = ensemble_row['PR-AUC']

st.markdown(f"""
### 🥇 Best Model: **XGBoost**
- **F1-Score:** {xgb_f1:.4f}  
- **PR-AUC:** {xgb_pr_auc:.4f}

XGBoost achieves the highest performance on both F1-Score and PR-AUC metrics on our current test set, making it the top performer for fraud detection with our existing data.
""")

st.markdown(f"""
### 🔄 Ensemble Consideration

While XGBoost is the best performer on our current data, the Stacking Ensemble also demonstrates strong capabilities:
- **F1-Score:** {ensemble_f1:.4f} (marginally lower than XGBoost)
- **PR-AUC:** {ensemble_pr_auc:.4f} (marginally lower than XGBoost)

**When to Consider the Ensemble:**

In real-world systems, fraud patterns evolve over time. Fraudsters constantly adapt their tactics, and transaction characteristics shift (amounts, timings, device patterns, merchant types). Here's why the ensemble can be valuable:

1. **Robustness to Pattern Changes:** XGBoost's split-based rules can become brittle when fraudsters change their behavior. The ensemble, by blending different modeling perspectives (linear, deep learning, gradient boosting), tends to be more stable when these shifts occur.

2. **Adaptive Defense:** As the statistical profile of transactions gradually shifts, an ensemble approach is more resilient because different models may capture different aspects of the evolved fraud patterns.

3. **Long-term Performance:** While XGBoost performs best today, the ensemble may offer better performance stability if fraud behavior starts evolving in ways the original model didn't anticipate.

**Recommendation:**
- **For immediate deployment:** Use XGBoost for maximum current accuracy
- **For production resilience:** Consider the ensemble to hedge against evolving fraud patterns and ensure more stable long-term performance
""")

st.markdown("""
---

## How to Read This Comparison

- **F1-Score:** Harmonic mean of precision and recall. Higher is better (0-1 scale)
- **PR-AUC:** Area under the Precision-Recall curve. Better for imbalanced datasets like fraud detection. Higher is better (0-1 scale)
- **Primary Metric:** PR-AUC is more meaningful for fraud detection since fraud cases are rare

**Key Takeaway:** Compare the metrics above to understand each model's strengths and choose based on your deployment constraints and long-term fraud pattern expectations.
""")


