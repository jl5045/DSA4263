import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="EDA Insights",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EDA Hypotheses: Visual Validation")

# --- Configuration ---
PLOTS_DIR = Path("plots")

# --- Main Page Logic ---
if not PLOTS_DIR.exists() or not any(PLOTS_DIR.iterdir()):
    st.error("Precomputed plot images not found!")
    st.info("Please run the precomputation script once from your terminal to generate the plots.")
    st.code("pip install -r requirements.txt\n./scripts/run_precompute_and_streamlit.sh", language="bash")
    st.stop()

st.success(f"Displaying precomputed plots from `{PLOTS_DIR}`.")
st.markdown("---")

# --- Helper to display plots ---
def display_plot(header, plot_filenames, conclusion_text, conclusion_status="success"):
    st.header(header)
    if isinstance(plot_filenames, str):
        plot_filenames = [plot_filenames]
    cols = st.columns(len(plot_filenames))
    for i, filename in enumerate(plot_filenames):
        plot_path = PLOTS_DIR / filename
        if plot_path.exists():
            try:
                image = Image.open(plot_path)
                with cols[i]:
                    st.image(image, use_column_width=True, caption=filename)
            except Exception as e:
                with cols[i]:
                    st.error(f"Could not load plot: {filename}. Error: {e}")
        else:
            with cols[i]:
                st.warning(f"Plot not found: {filename}")
    if conclusion_status == "success":
        st.success(conclusion_text)
    elif conclusion_status == "error":
        st.error(conclusion_text)
    else:
        st.info(conclusion_text)
    st.markdown("---")

# --- Display Hypotheses and Plots ---

# --- Display Hypotheses and Plots ---

display_plot(
    "Hypothesis: Fraud is often concentrated in TRANSFER and CASH_OUT.",
    ["eda_transaction_type_counts.png", "eda_fraud_rate_by_transaction_type.png"],
    "**Conclusion: True.** As shown in the Fraud Rate by Transaction Type Graph, fraud is indeed concentrated in `TRANSFER` and `CASH_OUT` transactions."
)

display_plot(
    "Hypothesis: Fraud transactions involve unusually high amounts to maximise gains.",
    "eda_transaction_amounts_fraud_vs_nonfraud.png",
    "**Conclusion: Partially True.** The boxplot shows that while non-fraudulent transactions have more extreme high-value outliers, the overall distribution for fraudulent transactions is shifted higher. This indicates that while not all high-value transactions are fraudulent, a fraudulent transaction is more likely to involve a large amount."
)

display_plot(
    "Hypothesis: A small number of senders are responsible for a large portion of transaction volume.",
    "eda_top10_senders_total_amount.png",
    "**Conclusion: True.** The bar chart shows the top 10 senders which arev involved in at least 1 fraud transaction by total transaction amount, highlighting that a few accounts move significant funds, which could be a useful feature for anomaly detection."
)

display_plot(
    "Hypothesis: Features exhibit multicollinearity, even with low pairwise correlation.",
    ["eda_correlation_heatmap.png", "eda_vif_collinearity.png"],
    "**Conclusion: True.** The correlation heatmap shows weak pairwise linear relationships. However, the VIF (Variance Inflation Factor) plot reveals moderate to high multicollinearity, indicating that some features are linearly predictable from a combination of others. This is critical for selecting the right modeling approach.",
    conclusion_status="info"
)
