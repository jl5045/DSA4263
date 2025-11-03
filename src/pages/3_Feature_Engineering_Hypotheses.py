import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Feature Engineering Insights",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Feature Engineering Hypotheses: Visual Validation")

# --- Configuration ---
PLOTS_DIR = Path("plots")

# --- Main Page Logic ---
if not PLOTS_DIR.exists() or not any(PLOTS_DIR.iterdir()):
    st.error("Precomputed plot images not found!")
    st.info("Please run the precomputation script or notebook once from your terminal to generate the plots.")
    st.code("pip install -r requirements.txt\n# Then run the 0_EDA.ipynb notebook", language="bash")
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
    "Hypothesis: Fraudulent users often execute many transactions in short time spans.",
    "Train_transaction_recency.png",
    "**Conclusion: False.** The plot shows that there is no significant difference in transaction recency between fraudulent and non-fraudulent accounts. This contradicts the idea that fraudsters transact in rapid bursts, as metrics like 'Percentage of Received Funds Forwarded Within 24 hours' would likely show no accounts forwarding funds that quickly.",
    conclusion_status="error"
)

display_plot(
    "Hypothesis: Fraudsters or mule accounts have many unique connections.",
    ["Train_pct_unique_senders.png", "Train_pct_unique_receivers.png"],
    "**Conclusion: False.** The `pctUniqueDest` plot shows a value near 100% for almost all users, both fraudulent and non-fraudulent. This means nearly every transaction goes to a new, unique recipient, regardless of legitimacy. This feature does not help distinguish fraudsters, who appear to use a 'hit-and-run' strategy rather than building a network of connections.",
    conclusion_status="error"
)

display_plot(
    "Hypothesis: Fraud involves the same two accounts repeatedly exchanging funds to launder money.",
    ["Train_pair_frequency_boxplot.png", "Train_pair_frequency_distribution.png"],
    "**Conclusion: False.** The graph shows the pair frequency for nearly all transactions is 1. This means the vast majority of sender-receiver pairs transact only once. Fraudulent users do not repeatedly target the same recipients; instead, the fraud is distributed in a 'hit-and-run' manner, targeting many different accounts once.",
    conclusion_status="error"
)
