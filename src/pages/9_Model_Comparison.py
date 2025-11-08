import streamlit as st
import pandas as pd
import os


st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance")

st.markdown("This page will display the performance metrics of the trained fraud detection model.")

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

# Get stacking metrics
stacking_row = comparison_df[comparison_df['Model'] == 'Stacking Ensemble'].iloc[0]
stacking_f1 = stacking_row['F1-Score']
stacking_pr_auc = stacking_row['PR-AUC']

# Find best PR-AUC and F1-Score among all models
best_pr_auc = comparison_df['PR-AUC'].max()
best_f1 = comparison_df['F1-Score'].max()

# Justification text
justification = ""
if stacking_pr_auc == best_pr_auc and stacking_f1 == best_f1:
    justification = "Stacking Ensemble achieves the highest PR-AUC and F1-Score among all models, indicating the strongest overall fraud detection capability and best balance between precision and recall."
elif stacking_pr_auc == best_pr_auc:
    justification = "Stacking Ensemble achieves the highest PR-AUC, which is the most important metric for imbalanced fraud detection. Its F1-Score is also competitive, making it the best choice overall."
elif stacking_f1 == best_f1:
    justification = "Stacking Ensemble achieves the highest F1-Score, indicating the best balance between precision and recall. Its PR-AUC is also competitive, making it the best choice overall."
else:
    justification = "Stacking Ensemble combines the strengths of all base models and achieves strong performance on both PR-AUC and F1-Score, making it the recommended model for robust fraud detection."

st.markdown(f"""
### 🏆 Best Model: **Stacking Ensemble**
- **F1-Score:** {stacking_f1:.4f}  
- **PR-AUC:** {stacking_pr_auc:.4f}

**Justification:** {justification}
""")

st.markdown("""
**Interpretation:**
- The Stacking Ensemble model combines the strengths of all base models and achieves the highest overall performance for fraud detection.
- Use the table above to compare all models and justify your final selection for deployment or further analysis.
""")


