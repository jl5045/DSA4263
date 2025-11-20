import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os

st.set_page_config(
    page_title="Network Explorer",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ Fraud Network Explorer")

st.markdown("""
This page visualizes the connections between accounts to reveal how fraudulent transactions cluster together. 
By exploring these networks, we can understand the patterns that the model uses to identify fraud.
""")

# --- Data Loading ---
# Note: Using downsampled train data for memory efficiency
DATASET_PATHS = {
    "Test without Merchants": "./data/FEwithoutMerchants/FE_test_without_merchants.csv",
    "Validation without Merchants": "./data/FEwithoutMerchants/FE_validation_without_merchants.csv",
    "Train (Downsampled 1:5) without Merchants": "./data/FEwithoutMerchants/FE_train_downsampled_1to5_without_merchants.csv",
    "Train (Downsampled 1:10) without Merchants": "./data/FEwithoutMerchants/FE_train_downsampled_1to10_without_merchants.csv",
    "Test with Merchants": "./data/FEwithMerchants/FE_test_with_merchants.csv",
    "Validation with Merchants": "./data/FEwithMerchants/FE_validation_with_merchants.csv",
    "Train (Downsampled 1:5) with Merchants": "./data/FEwithMerchants/FE_train_downsampled_1to5_with_merchants.csv",
    "Train (Downsampled 1:10) with Merchants": "./data/FEwithMerchants/FE_train_downsampled_1to10_with_merchants.csv",
}

def load_data(path: str):
    """Loads the specified feature-engineered dataset (no caching to prevent memory issues)."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Engineered data not found at `{path}`. Please run the feature engineering pipeline first.")
        return None

st.sidebar.header("Dataset Selection")
selected_dataset_name = st.sidebar.selectbox(
    "Choose a dataset to explore:",
    list(DATASET_PATHS.keys())
)

# Clear cache button to manually free memory
if st.sidebar.button("🗑️ Clear Memory Cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Switch datasets to free memory.")
    st.rerun()

st.sidebar.info("💡 Tip: Each dataset is ~300MB. If the app crashes, click 'Clear Memory Cache' before loading a new dataset.")

with st.spinner(f"Loading {selected_dataset_name}..."):
    df = load_data(DATASET_PATHS[selected_dataset_name])

if df is not None:
    st.info(f"Currently exploring the **{selected_dataset_name}** dataset with **{len(df):,}** transactions.")
    # --- Interactive Fraud Cluster Selection ---
    all_node_degrees = pd.concat([df['nameOrig'], df['nameDest']]).value_counts()

    # --- Interactive Fraud Cluster Selection ---
    fraudulent_transactions = df[df['isFraud'] == 1].copy()
    if fraudulent_transactions.empty:
        st.warning("No fraudulent transactions found in the loaded data.")
    else:
        # Calculate a connectivity score for each fraud transaction
        fraudulent_transactions['sender_degree'] = fraudulent_transactions['nameOrig'].map(all_node_degrees).fillna(0)
        fraudulent_transactions['receiver_degree'] = fraudulent_transactions['nameDest'].map(all_node_degrees).fillna(0)
        fraudulent_transactions['connectivity_score'] = fraudulent_transactions['sender_degree'] + fraudulent_transactions['receiver_degree']

        # Sort the fraudulent transactions by the new score
        sorted_frauds = fraudulent_transactions.sort_values('connectivity_score', ascending=False)

        st.sidebar.header("Network Controls")
        selected_fraud_index = st.sidebar.selectbox(
            "Select a fraudulent transaction (ranked by cluster size):",
            sorted_frauds.index,
            format_func=lambda x: f"Tx {x} (Connections: {int(sorted_frauds.loc[x, 'connectivity_score'])}) | {df.loc[x, 'nameOrig']} -> {df.loc[x, 'nameDest']}"
        )

        fraud_transaction = df.loc[selected_fraud_index]
        sender = fraud_transaction['nameOrig']
        receiver = fraud_transaction['nameDest']

        st.write(f"Exploring the network around the fraudulent transaction: **{sender} → {receiver}**")

        # --- Build the Local Network Graph ---
        # Find all accounts connected to the sender and receiver (1st-degree neighbors)
        first_degree_nodes = set()
        first_degree_nodes.update(df[df['nameOrig'] == sender]['nameDest'])
        first_degree_nodes.update(df[df['nameDest'] == sender]['nameOrig'])
        first_degree_nodes.update(df[df['nameOrig'] == receiver]['nameDest'])
        first_degree_nodes.update(df[df['nameDest'] == receiver]['nameOrig'])
        
        # The full neighborhood includes the original sender/receiver and their direct connections
        neighborhood = first_degree_nodes.union({sender, receiver})

        # Filter the dataframe to get all transactions *within* this neighborhood
        local_network_df = df[
            df['nameOrig'].isin(neighborhood) & df['nameDest'].isin(neighborhood)
        ]

        st.write(f"Found **{len(local_network_df)}** transactions within the 1st-degree neighborhood of the fraudulent parties.")

        G = nx.from_pandas_edgelist(
            local_network_df,
            source='nameOrig',
            target='nameDest',
            edge_attr='amount',
            create_using=nx.DiGraph()
        )

        # --- Degree Analysis ---
        if G.number_of_nodes() > 0:
            degrees = G.degree()
            in_degrees = G.in_degree()
            out_degrees = G.out_degree()

            degree_df = pd.DataFrame({
                'Account': [node for node, __ in degrees],
                'Total Connections (Degree)': [deg for __, deg in degrees],
                'Incoming Transactions (In-Degree)': [deg for __, deg in in_degrees],
                'Outgoing Transactions (Out-Degree)': [deg for __, deg in out_degrees],
            }).sort_values('Total Connections (Degree)', ascending=False).reset_index(drop=True)

            st.subheader("Most Connected Accounts in Cluster")
            st.dataframe(degree_df)
        else:
            degree_df = pd.DataFrame()

        # --- Create Pyvis Visualization ---
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)

        # Set physics layout for better visualization
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -80000,
              "springConstant": 0.001,
              "springLength": 200
            },
            "minVelocity": 0.75
          }
        }
        """)

        # Add nodes and edges to the pyvis network
        for node in G.nodes():
            if node == sender:
                color = "orange"
                title_prefix = "Fraudulent Sender"
            elif node == receiver:
                color = "red"
                title_prefix = "Fraudulent Receiver"
            else:
                color = "lightgreen"
                title_prefix = "Connected Account"
            
            # Get feature info for the hover-over title and node size
            try:
                node_features = df[df['nameOrig'] == node].iloc[0]
                pagerank = node_features.get('sender_pr', 0)
                out_degree = node_features.get('sender_outdeg_cnt', 0)
                in_degree = node_features.get('sender_indeg_cnt', 0)
                
                title = (
                    f"{title_prefix}: {node}<br>"
                    f"PageRank: {pagerank:.4f}<br>"
                    f"Outgoing Connections: {int(out_degree)}<br>"
                    f"Incoming Connections: {int(in_degree)}"
                )
                # Scale node size by PageRank
                size = 15 + pagerank * 1000
            except (IndexError, KeyError): # Fallback if node not found as a sender
                pagerank = df[df['nameDest'] == node].iloc[0].get('receiver_pr', 0)
                title = f"{title_prefix}: {node}<br>PageRank: {pagerank:.4f}"
                size = 15 + pagerank * 1000

            net.add_node(node, title=title, color=color, size=size)

        for edge in G.edges(data=True):
            src, dst, data = edge
            net.add_edge(src, dst, value=data['amount'], title=f"Amount: {data['amount']:.2f}")

        # --- Save and Display the Graph ---
        try:
            path = './plots'
            if not os.path.exists(path):
                os.makedirs(path)
            file_path = os.path.join(path, "fraud_network.html")
            net.save_graph(file_path)
            
            # Embed the HTML file in Streamlit
            with open(file_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=800)

        except Exception as e:
            st.error(f"Could not generate or display the network graph: {e}")
