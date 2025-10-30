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
PLOTS_DIR = Path("data/processed/eda/plots")

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

display_plot(
    "Hypothesis: Fraud is often concentrated in TRANSFER and CASH_OUT.",
    "1_fraud_rate_by_type.png",
    "**Conclusion: True.** As shown in the Fraud Rate by Transaction Type Graph in the EDA, fraud is indeed concentrated in `TRANSFER` and `CASH_OUT` transactions."
)

display_plot(
    "Hypothesis: Fraud transactions involve unusually high amounts to maximise gains.",
    ["2_amount_distribution.png", "3_fraud_prob_by_amount.png"],
    "**Conclusion: True.** While non-fraud has more outliers in the simple distribution, the Fraud Probability Across Transactions Amount Ranges plot shows that fraud is indeed most probable at the highest amounts."
)

display_plot(
    "Hypothesis: Accounts exhibiting unusually high hourly averages are often correlated with fraud.",
    "8_hourly_averages.png",
    "**Conclusion: True.** The box plot shows that the distribution of average hourly transaction amounts is noticeably higher for fraudulent senders, supporting the hypothesis."
)

display_plot(
    "Hypothesis: Fraudulent users often execute many transactions in short time spans.",
    "4_transaction_recency.png",
    "**Conclusion: False.** The Transaction Recency plot shows that the distribution of time between transactions is very similar for both fraudulent and non-fraudulent users. This contradicts the idea that fraudsters transact in rapid bursts.",
    conclusion_status="error"
)

display_plot(
    "Hypothesis: Fraud follows predictable patterns like `TRANSFER` -> `CASH_OUT`.",
    "5_sequence_pattern.png",
    "**Conclusion: True.** The bar chart shows a very high occurrence of the `TRANSFER` -> `CASH_OUT` sequence in fraudulent transactions compared to non-fraudulent ones."
)

display_plot(
    "Hypothesis: Fraudsters have many unique connections, unlike regular users.",
    "6_pct_unique_dest.png",
    "**Conclusion: False.** The `pctUniqueDest` plot shows that almost every sender, fraudulent or not, interacts with a unique receiver each time. A value of 100% is the norm for everyone, making this feature not useful for distinguishing fraud.",
    conclusion_status="error"
)

display_plot(
    "Hypothesis: Fraud involves the same two accounts repeatedly exchanging funds.",
    "7_pair_frequency.png",
    "**Conclusion: False.** The graph shows that the pair frequency for almost all transactions is 1. This indicates that the vast majority of sender-receiver pairs transact only once, meaning fraudulent users do not repeatedly target the same recipient. The fraud is distributed or 'hit-and-run'.",
    conclusion_status="error"
)
