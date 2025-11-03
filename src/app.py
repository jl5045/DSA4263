import streamlit as st

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("Welcome to the Financial Fraud Detection App!")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This interactive application is designed to showcase the results of a machine learning model
    trained to detect fraudulent financial transactions.

    **👈 Select a page from the sidebar** to explore different aspects of the project.
    """
)
