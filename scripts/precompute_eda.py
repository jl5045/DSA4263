"""
Precompute EDA artifacts from the raw synthetic financial dataset by streaming.
This script processes the entire dataset in chunks to remain memory-efficient,
generating static PNG plots for all key analyses found in the user's notebooks.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def ensure_out(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

def save_plot(fig_or_ax, path: Path):
    try:
        fig = fig_or_ax.get_figure() if isinstance(fig_or_ax, plt.Axes) else fig_or_ax
        fig.savefig(path.with_suffix('.png'))
        plt.close(fig)
        print(f"Saved plot: {path.with_suffix('.png')}")
    except Exception as e:
        print(f"Could not save plot to {path}. Error: {e}")
        print("Please ensure you have 'matplotlib' and 'seaborn' installed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to raw CSV')
    parser.add_argument('--outdir', type=str, default='data/processed/eda', help='Output directory for artifacts')
    parser.add_argument('--chunksize', type=int, default=500000, help='Rows to process per chunk')
    args = parser.parse_args()

    inp = Path(args.input)
    assert inp.exists(), f"Input {inp} does not exist"
    outdir = Path(args.outdir)
    plots_dir = outdir / 'plots'
    ensure_out(plots_dir)

    # --- PASS 1: Aggregate all necessary data by streaming through the file ---
    print("--- Starting Pass 1: Aggregating data from the entire dataset... ---")
    
    # Initialize accumulators
    type_counts = Counter()
    type_fraud_counts = Counter()
    pair_counts = Counter()
    receiver_type_sets = defaultdict(set)
    receiver_stats_agg = defaultdict(lambda: (0, 0.0, 0.0))
    type_stats_agg = defaultdict(lambda: (0, 0.0, 0.0))
    amount_samples_fraud = []
    amount_samples_nonfraud = []
    sample_size = 50000
    last_event = {}
    sequences = []
    # For hourly averages
    hourly_amounts = defaultdict(list)
    # For recency
    last_step = {}
    recency_fraud = []
    recency_nonfraud = []
    # For unique connections
    sender_total_counts = Counter()
    sender_unique_dests = defaultdict(set)

    total_rows = 0
    for chunk in pd.read_csv(inp, chunksize=args.chunksize):
        total_rows += len(chunk)
        print(f"  Processing rows {total_rows - len(chunk):,} to {total_rows:,}...")
        # Fraud by type
        for _, row in chunk.iterrows():
            type_counts[row['type']] += 1
            if row['isFraud']:
                type_fraud_counts[row['type']] += 1
            pair_counts[(row['nameOrig'], row['nameDest'])] += 1
            receiver_type_sets[row['nameDest']].add(row['type'])
            # Welford for receiver/type stats
            def update_welford(agg, value):
                count, mean, M2 = agg
                count += 1
                delta = value - mean
                mean += delta / count
                delta2 = value - mean
                M2 += delta * delta2
                return (count, mean, M2)
            receiver_stats_agg[row['nameDest']] = update_welford(receiver_stats_agg[row['nameDest']], row['amount'])
            type_stats_agg[row['type']] = update_welford(type_stats_agg[row['type']], row['amount'])
            # Sequence pattern
            sender = row['nameOrig']
            if sender in last_event:
                prev_row = last_event[sender]
                if prev_row['type'] == 'TRANSFER' and row['type'] == 'CASH_OUT':
                    sequences.append(row)
            last_event[sender] = row
            # Hourly averages
            hour = int(row['step']) % 24
            hourly_amounts[hour].append(row['amount'])
            # Unique connections
            sender_total_counts[row['nameOrig']] += 1
            sender_unique_dests[row['nameOrig']].add(row['nameDest'])

        # Amount samples
        fraud_amounts = chunk[chunk['isFraud'] == 1]['amount']
        nonfraud_amounts = chunk[chunk['isFraud'] == 0]['amount']
        for val in fraud_amounts:
            if len(amount_samples_fraud) < sample_size:
                amount_samples_fraud.append(val)
        for val in nonfraud_amounts:
            if len(amount_samples_nonfraud) < sample_size:
                amount_samples_nonfraud.append(val)

    print("--- Pass 1 Finished. Finalizing initial dataframes... ---")

    fraud_by_type = pd.DataFrame({
        'type': list(type_counts.keys()),
        'total': [type_counts[t] for t in type_counts.keys()],
        'fraud': [type_fraud_counts.get(t, 0) for t in type_counts.keys()]
    })
    fraud_by_type['fraud_rate'] = fraud_by_type['fraud'] / fraud_by_type['total']
    
    amount_samples_df = pd.DataFrame({
        'amount': amount_samples_fraud + amount_samples_nonfraud,
        'isFraud': [1] * len(amount_samples_fraud) + [0] * len(amount_samples_nonfraud)
    })
    
    receiver_stats_list = []
    for dest, (count, mean, M2) in receiver_stats_agg.items():
        variance = M2 / (count - 1) if count > 1 else 0
        std = np.sqrt(variance)
        receiver_stats_list.append({'nameDest': dest, 'amount_mean': mean, 'amount_std': std})
    receiver_stats = pd.DataFrame(receiver_stats_list)
    
    type_stats_list = []
    for t, (count, mean, M2) in type_stats_agg.items():
        variance = M2 / (count - 1) if count > 1 else 0
        std = np.sqrt(variance)
        type_stats_list.append({'type': t, 'mean': mean, 'std': std})
    type_stats = pd.DataFrame(type_stats_list)
    
    pair_counts_df = pd.DataFrame(pair_counts.items(), columns=['pair', 'count'])
    pair_counts_df[['nameOrig', 'nameDest']] = pd.DataFrame(pair_counts_df['pair'].tolist(), index=pair_counts_df.index)
    
    receiver_types_df = pd.DataFrame([{'nameDest': k, 'num_types': len(v)} for k, v in receiver_type_sets.items()])

    # --- PASS 2: Calculate stats that required Pass 1 results ---
    print("\n--- Starting Pass 2 of 2: Calculating ratio and binned statistics... ---")
    _, amount_bins = pd.qcut(amount_samples_df['amount'], q=10, retbins=True, duplicates='drop')
    type_mean_map = type_stats.set_index('type')['mean'].to_dict()

    bin_counts = Counter()
    bin_fraud_counts = Counter()
    amount_ratio_fraud = []
    amount_ratio_nonfraud = []

    total_rows = 0
    for chunk in pd.read_csv(inp, chunksize=args.chunksize):
        total_rows += len(chunk)
        print(f"  Pass 2: Processing rows {total_rows - len(chunk):,} to {total_rows:,}...")
        
        chunk['amount_bin'] = pd.cut(chunk['amount'], bins=amount_bins, labels=False, include_lowest=True)
        chunk['amountTypeRatio'] = chunk['amount'] / chunk['type'].map(type_mean_map).fillna(1)

        for _, row in chunk.iterrows():
            if pd.notna(row['amount_bin']):
                bin_counts[row['amount_bin']] += 1
                if row['isFraud']:
                    bin_fraud_counts[row['amount_bin']] += 1
        
        fraud_ratios = chunk[chunk['isFraud'] == 1]['amountTypeRatio'].dropna()
        nonfraud_ratios = chunk[chunk['isFraud'] == 0]['amountTypeRatio'].dropna()

        for val in fraud_ratios:
            if len(amount_ratio_fraud) < sample_size:
                amount_ratio_fraud.append(val)
            else:
                idx = np.random.randint(0, total_rows)
                if idx < sample_size:
                    amount_ratio_fraud[idx] = val
        
        for val in nonfraud_ratios:
            if len(amount_ratio_nonfraud) < sample_size:
                amount_ratio_nonfraud.append(val)
            else:
                idx = np.random.randint(0, total_rows)
                if idx < sample_size:
                    amount_ratio_nonfraud[idx] = val

    print("--- Pass 2 Finished. Finalizing all plots... ---")

    prob_by_amount_list = []
    for b, total in bin_counts.items():
        fraud = bin_fraud_counts.get(b, 0)
        if total > 0:
            prob_by_amount_list.append({'amount_bin': b, 'fraud_rate': fraud / total})
    prob_by_amount = pd.DataFrame(prob_by_amount_list)
    
    amount_ratio_df = pd.DataFrame({
        'amountTypeRatio': amount_ratio_fraud + amount_ratio_nonfraud,
        'isFraud': [1] * len(amount_ratio_fraud) + [0] * len(amount_ratio_nonfraud)
    })

    # --- Generate and Save All Plots ---
    # Plot 1: Fraud Rate by Transaction Type
    import plotly.express as px
    fig1 = px.bar(fraud_by_type.sort_values('fraud_rate', ascending=False),
                  x='type', y='fraud_rate', title='Fraud Rate by Transaction Type')
    save_plot(fig1, plots_dir / 'H1_fraud_rate_by_type.png')

    # Plot 2: Amount Distribution (Sampled): Fraud vs Non-Fraud
    amount_samples_df['label'] = amount_samples_df['isFraud'].map({0: 'Non-Fraud', 1: 'Fraud'})
    fig2a = px.box(amount_samples_df, x='label', y='amount', log_y=True, title='Amount Distribution (Sampled): Fraud vs Non-Fraud')
    save_plot(fig2a, plots_dir / 'H2a_amount_comparison_box.png')

    # Plot 3: Fraud Probability by Amount Quantile
    bin_labels = [f"[{amount_bins[i]:.0f}-{amount_bins[i+1]:.0f}]" for i in range(len(amount_bins)-1)]
    prob_by_amount['bin_label'] = prob_by_amount['amount_bin'].map(dict(enumerate(bin_labels)))
    fig3 = px.bar(prob_by_amount, x='bin_label', y='fraud_rate', title='Fraud Probability by Amount Quantile')
    save_plot(fig3, plots_dir / 'H2b_fraud_prob_by_amount.png')

    # Plot 4: Transaction Recency Distribution by Fraud Label
    # Compute transaction recency for each transaction
    recency_list = []
    is_fraud_list = []
    last_step_map = {}
    for chunk in pd.read_csv(inp, chunksize=args.chunksize, usecols=['nameOrig', 'step', 'isFraud']):
        chunk = chunk.sort_values(['nameOrig', 'step'])
        for _, row in chunk.iterrows():
            sender = row['nameOrig']
            step = row['step']
            if sender in last_step_map:
                recency = step - last_step_map[sender]
                recency_list.append(recency)
                is_fraud_list.append(row['isFraud'])
            last_step_map[sender] = step
    recency_df = pd.DataFrame({
        'transactionRecency': recency_list,
        'isFraud': is_fraud_list
    })
    fig4 = plt.figure(figsize=(8, 5))
    sns.kdeplot(recency_df[recency_df['isFraud']==0]['transactionRecency'], label='Non-Fraud', fill=True)
    sns.kdeplot(recency_df[recency_df['isFraud']==1]['transactionRecency'], label='Fraud', fill=True)
    plt.title('Transaction Recency Distribution by Fraud Label')
    plt.xlabel('Steps Since Last Transaction')
    plt.legend()
    save_plot(fig4, plots_dir / 'H4_transaction_recency.png')

    # Plot 5: Distribution of Average Hourly Transaction Amounts
    hourly_avg = {h: np.mean(hourly_amounts[h]) if len(hourly_amounts[h]) > 0 else 0 for h in range(24)}
    hourly_df = pd.DataFrame({'hour': list(hourly_avg.keys()), 'avg_amount': list(hourly_avg.values())})
    fig5 = px.bar(hourly_df, x='hour', y='avg_amount', title='Distribution of Average Hourly Transaction Amounts')
    save_plot(fig5, plots_dir / 'H5_hourly_avg_amount.png')

    # Plot 6: % of Unique Receivers per Sender
    sender_stats = pd.DataFrame({
        'nameOrig': list(sender_total_counts.keys()),
        'totalSent': [sender_total_counts[s] for s in sender_total_counts.keys()],
        'numUniqueDest': [len(sender_unique_dests[s]) for s in sender_total_counts.keys()]
    })
    sender_stats['pctUniqueDest'] = (sender_stats['numUniqueDest'] / sender_stats['totalSent']) * 100
    fig6 = plt.figure(figsize=(8, 5))
    sns.histplot(sender_stats['pctUniqueDest'], bins=40, kde=True)
    plt.title('% of Unique Receivers per Sender')
    plt.xlabel('pctUniqueDest')
    save_plot(fig6, plots_dir / 'H6_pct_unique_dest.png')

    # Plot 7: Histogram of Transactions per Sender-Receiver Pair
    fig7 = px.histogram(pair_counts_df, x='count', title='Histogram of Transactions per Sender-Receiver Pair')
    fig7.update_xaxes(range=[0, 10])
    save_plot(fig7, plots_dir / 'H7_pair_frequency.png')

    print("\nPrecomputation of all plots from the full dataset is complete.")

if __name__ == '__main__':
    main()
