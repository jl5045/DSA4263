"""Reusable feature-engineering helpers for the FIN-FRAUD splits."""

from __future__ import annotations


import os
from typing import Callable, Dict, Iterable, List, MutableMapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:  # Notebook runtime will have display, but set a safe fallback.
    from IPython.display import display
except ImportError:  # pragma: no cover - fallback for pure python runs
    def display(obj):  # type: ignore
        print(obj)


SplitMap = MutableMapping[str, pd.DataFrame]
FeatureStep = Tuple[Callable[..., pd.DataFrame], Dict]


def create_split_frames(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
) -> SplitMap:
    """Return copies of every split so downstream steps remain side-effect free."""

    return {
        "Train": df_train.copy(),
        "Test": df_test.copy(),
        "Validation": df_val.copy(),
    }


def apply_to_splits(
    split_frames: SplitMap,
    func: Callable[..., pd.DataFrame],
    **kwargs,
) -> SplitMap:
    """Run *func* for every split and store the returned dataframe back in-place."""

    for split_name, split_df in split_frames.items():
        print(f"\n>>> Running {func.__name__} for {split_name}")
        split_frames[split_name] = func(split_df, split_name, **kwargs)
    return split_frames


def transaction_velocity(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Plot daily transaction velocity for the requested dataframe."""

    df['day'] = (df['step'] // 24).astype(int)
    fraud_daily = df[df['isFraud'] == 1].groupby('day').size().reset_index(name='fraud_count')
    nonfraud_daily = df[df['isFraud'] == 0].groupby('day').size().reset_index(name='nonfraud_count')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(
        fraud_daily['day'],
        fraud_daily['fraud_count'],
        color='red',
        marker='o',
        linewidth=2,
        markersize=6,
    )
    axes[0, 0].fill_between(
        fraud_daily['day'], fraud_daily['fraud_count'], alpha=0.3, color='red'
    )
    axes[0, 0].set_xlabel('Day', fontsize=12)
    axes[0, 0].set_ylabel('Fraud Transaction Count', fontsize=12)
    axes[0, 0].set_title(f'{split_name}: Daily Fraud Transactions (Zoomed)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        nonfraud_daily['day'],
        nonfraud_daily['nonfraud_count'],
        color='blue',
        marker='o',
        linewidth=2,
        markersize=6,
    )
    axes[0, 1].fill_between(
        nonfraud_daily['day'], nonfraud_daily['nonfraud_count'], alpha=0.3, color='blue'
    )
    axes[0, 1].set_xlabel('Day', fontsize=12)
    axes[0, 1].set_ylabel('Non-Fraud Transaction Count', fontsize=12)
    axes[0, 1].set_title(f'{split_name}: Daily Non-Fraud Transactions', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    line1 = ax1.plot(
        fraud_daily['day'],
        fraud_daily['fraud_count'],
        color='red',
        marker='o',
        linewidth=2,
        markersize=6,
        label='Fraud',
    )
    line2 = ax2.plot(
        nonfraud_daily['day'],
        nonfraud_daily['nonfraud_count'],
        color='blue',
        marker='o',
        linewidth=2,
        markersize=6,
        label='Non-Fraud',
    )
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Fraud Count', fontsize=12, color='red')
    ax2.set_ylabel('Non-Fraud Count', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'{split_name}: Daily Transactions (Dual Axis)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    fraud_rate_daily = fraud_daily.copy()
    fraud_rate_daily['total'] = (
        fraud_daily['fraud_count'] + nonfraud_daily['nonfraud_count']
    ).replace(0, np.nan)
    fraud_rate_daily['fraud_rate'] = fraud_daily['fraud_count'] / fraud_rate_daily['total']

    axes[1, 1].plot(
        fraud_rate_daily['day'],
        fraud_rate_daily['fraud_rate'],
        color='purple',
        marker='o',
        linewidth=2,
        markersize=6,
    )
    axes[1, 1].fill_between(
        fraud_rate_daily['day'],
        fraud_rate_daily['fraud_rate'],
        alpha=0.3,
        color='purple',
    )
    axes[1, 1].set_xlabel('Day', fontsize=12)
    axes[1, 1].set_ylabel('Fraud Rate', fontsize=12)
    axes[1, 1].set_title(f'{split_name}: Daily Fraud Rate', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))

    fig.suptitle(f'Transaction Velocity Analysis for {split_name} Data', fontsize=16)
    plt.tight_layout()
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_transaction_velocity.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    return df


def add_avg_amount_features(
    df: pd.DataFrame,
    split_name: str,
    quantile_clip: float = 0.95,
    plots_dir: str | None = None,
) -> pd.DataFrame:
    """Create avgAmountToDest and visualise distributions by label."""

    avg_amount_to_dest = (
        df.groupby(['nameOrig', 'nameDest'])['amount']
        .mean()
        .reset_index(name='avgAmountToDest')
    )
    df = df.merge(avg_amount_to_dest, on=['nameOrig', 'nameDest'], how='left')
    print(f"{split_name}: avgAmountToDest statistics")
    print(df.groupby('isFraud')['avgAmountToDest'].describe())
    print(df[['nameOrig', 'nameDest', 'amount', 'avgAmountToDest', 'isFraud']].head(10))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    nonfraud = df[df['isFraud'] == 0]['avgAmountToDest'].dropna()
    fraud = df[df['isFraud'] == 1]['avgAmountToDest'].dropna()
    axes[0].hist(nonfraud, bins=50, alpha=0.6, label='Non-Fraud', color='blue', density=True)
    axes[0].hist(fraud, bins=50, alpha=0.6, label='Fraud', color='red', density=True)
    axes[0].set_xlabel('Average Amount to Destination', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'{split_name}: Avg Amount to Destination Distribution', fontsize=14)
    axes[0].legend()
    upper = df['avgAmountToDest'].dropna().quantile(quantile_clip)
    if np.isfinite(upper):
        axes[0].set_xlim(0, upper)

    axes[1].boxplot([nonfraud, fraud], labels=['Non-Fraud', 'Fraud'])
    axes[1].set_ylabel('Average Amount to Destination', fontsize=12)
    axes[1].set_title(f'{split_name}: Avg Amount vs Fraud Label', fontsize=14)
    axes[1].set_yscale('log')
    plt.tight_layout()
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_avg_amount_features.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    return df


def add_receiver_flow_features(
    df: pd.DataFrame,
    split_name: str,
    top_n: int = 10,
    plots_dir: str | None = None,
) -> pd.DataFrame:
    """Add receiver volatility stats and inspect the top irregular accounts."""

    receiver_amt_stats = (
        df.groupby('nameDest')['amount']
        .agg(['max', 'std'])
        .reset_index()
        .rename(columns={'max': 'maxAmountReceived', 'std': 'stdAmountReceived'})
    )
    df = df.merge(receiver_amt_stats, on='nameDest', how='left')
    df['std_to_mean_ratio'] = df['stdAmountReceived'] / (df['avgAmountToDest'] + 1e-9)
    top_irregular = (
        df.groupby('nameDest')['std_to_mean_ratio']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    if top_irregular.empty:
        print(f"No receivers found for irregularity plot in {split_name}.")
        return df

    fig = top_irregular.plot(
        kind='bar', figsize=(8, 5), title=f'{split_name}: Top Receivers by Inflow Irregularity'
    ).get_figure()
    plt.ylabel('std_to_mean_ratio')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_receiver_flow_features.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()

    for receiver in top_irregular.index:
        receiver_txns = df[df['nameDest'] == receiver]
        fraud_count = receiver_txns['isFraud'].sum()
        total = len(receiver_txns)
        rate = (fraud_count / total * 100) if total else 0.0
        print(
            f"Receiver: {receiver} | transactions: {total} | fraud: {fraud_count} |"
            f" fraud rate: {rate:.2f}%"
        )
    return df


def plot_receiver_type_heatmap(
    df: pd.DataFrame,
    split_name: str,
    top_n: int = 15,
    plots_dir: str | None = None,
) -> pd.DataFrame:
    """Plot a receiver vs transaction-type heatmap for the busiest accounts."""

    top_receivers = df['nameDest'].value_counts().nlargest(top_n).index
    subset = df[df['nameDest'].isin(top_receivers)]
    if subset.empty:
        print(f"No receiver/type combinations available for {split_name}.")
        return df

    heatmap_data = pd.crosstab(subset['nameDest'], subset['type'])
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap='Blues', annot=True, fmt='d')
    plt.title(f'{split_name}: Receiver–Transaction Type Frequency')
    plt.xlabel('Transaction Type')
    plt.ylabel('Receiver (nameDest)')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_receiver_type_heatmap.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    return df


def add_transaction_type_features(
    df: pd.DataFrame,
    split_name: str,
    sample_size: int = 2_000_000,
    plots_dir: str | None = None,
) -> pd.DataFrame:
    """Create amount/type features and associated visuals."""

    type_amt_stats = (
        df.groupby('type')['amount']
        .agg(['mean', lambda x: x.quantile(0.95)])
        .reset_index()
        .rename(columns={'mean': 'avgAmountPerType', '<lambda_0>': 'p95AmountPerType'})
    )
    df = df.merge(type_amt_stats, on='type', how='left')
    df['amountTypeRatio'] = df['amount'] / (df['avgAmountPerType'] + 1e-9)
    df['typeHighValueFlag'] = (df['amount'] > df['p95AmountPerType']).astype(int)

    type_summary = (
        df.groupby('type')['amount']
        .agg(['mean', 'std'])
        .sort_values(by='mean', ascending=False)
    )
    fig1 = plt.figure(figsize=(8, 5))
    type_summary['mean'].plot(kind='bar', yerr=type_summary['std'], capsize=4)
    plt.title(f'{split_name}: Avg Transaction Amount and Variability by Type')
    plt.ylabel('Average Amount')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_transaction_type_variability.png")
        fig1.savefig(plot_path)
        plt.close(fig1)
    else:
        plt.show()

    sample_n = min(sample_size, len(df))
    if sample_n > 0:
        sample_df = df.sample(sample_n, random_state=42)
        fig2 = plt.figure(figsize=(10, 6))
        sns.stripplot(data=sample_df, x='type', y='amount', hue='isFraud', alpha=0.5)
        plt.yscale('log')
        plt.title(f'{split_name}: Transaction Amounts by Type (Fraud highlighted)')
        plt.xlabel('Transaction Type')
        plt.ylabel('Amount (log scale)')
        plt.legend(title='Fraudulent?')
        
        if plots_dir:
            plot_path = os.path.join(plots_dir, f"{split_name}_transaction_type_stripplot.png")
            fig2.savefig(plot_path)
            plt.close(fig2)
        else:
            plt.show()

    df['amountLog'] = np.log1p(df['amount'])
    df['amountBucket'] = pd.qcut(df['amount'], 10, duplicates='drop')
    fraud_rate_by_bin = (
        df.groupby('amountBucket')['isFraud'].mean().reset_index(name='fraudProbabilityByAmountBin'
        )
    )
    df = df.merge(fraud_rate_by_bin, on='amountBucket', how='left')

    fig3 = plt.figure(figsize=(8, 5))
    sns.barplot(
        data=fraud_rate_by_bin,
        x='amountBucket',
        y='fraudProbabilityByAmountBin',
        color='orange',
    )
    plt.title(f'{split_name}: Fraud Probability Across Amount Ranges')
    plt.xlabel('Transaction Amount Range')
    plt.ylabel('Fraud Probability')
    plt.xticks(rotation=45)
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_fraud_prob_by_amount.png")
        fig3.savefig(plot_path)
        plt.close(fig3)
    else:
        plt.show()

    if len(df):
        df_sorted = df.sort_values('amount')
        fraud_total = df_sorted['isFraud'].sum() or 1
        cum_fraud = df_sorted['isFraud'].cumsum() / fraud_total
        cum_trans = np.arange(len(df_sorted)) / len(df_sorted)
        fig4 = plt.figure(figsize=(8, 5))
        plt.plot(df_sorted['amount'], cum_fraud, label='Cumulative Fraud')
        plt.plot(
            df_sorted['amount'],
            cum_trans,
            label='Cumulative Transactions',
            linestyle='--',
        )
        plt.title(f'{split_name}: Cumulative Fraud vs Total Transactions')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Cumulative Share')
        plt.legend()
        plt.xlim(0, df['amount'].quantile(0.99))
        
        if plots_dir:
            plot_path = os.path.join(plots_dir, f"{split_name}_cumulative_fraud.png")
            fig4.savefig(plot_path)
            plt.close(fig4)
        else:
            plt.show()

    return df


def add_sender_receiver_aggregations(df: pd.DataFrame, split_name: str, **kwargs) -> pd.DataFrame:
    """Aggregate sender/receiver statistics and merge them back."""

    sender_agg = df.groupby('nameOrig').agg({
        'amount': ['sum', 'mean', 'std', 'count'],
        'isFraud': 'max',
    }).reset_index()
    sender_agg.columns = ['nameOrig', 'totalSent', 'meanSent', 'stdSent', 'numSent', 'sender_has_fraud']

    type_fractions = df.groupby(['nameOrig', 'type']).size().unstack(fill_value=0)
    type_fractions = type_fractions.div(type_fractions.sum(axis=1), axis=0)
    type_fractions.columns = [f'fraction_{col}' for col in type_fractions.columns]
    type_fractions = type_fractions.reset_index()
    sender_agg = sender_agg.merge(type_fractions, on='nameOrig', how='left')

    receiver_agg = df.groupby('nameDest').agg({
        'amount': ['sum', 'mean', 'std', 'count'],
        'isFraud': 'max',
    }).reset_index()
    receiver_agg.columns = ['nameDest', 'totalReceived', 'meanReceived', 'stdReceived', 'numReceived', 'receiver_has_fraud']

    type_fractions_recv = df.groupby(['nameDest', 'type']).size().unstack(fill_value=0)
    type_fractions_recv = type_fractions_recv.div(type_fractions_recv.sum(axis=1), axis=0)
    type_fractions_recv.columns = [f'fraction_{col}_recv' for col in type_fractions_recv.columns]
    type_fractions_recv = type_fractions_recv.reset_index()
    receiver_agg = receiver_agg.merge(type_fractions_recv, on='nameDest', how='left')

    df = df.merge(sender_agg, on='nameOrig', how='left')
    df = df.merge(receiver_agg, on='nameDest', how='left')

    sender_neighbor = df.groupby('nameOrig').agg({
        'nameDest': 'nunique',
        'receiver_has_fraud': 'sum',
    }).reset_index()
    sender_neighbor.columns = ['nameOrig', 'num_receivers', 'num_fraud_receivers']
    sender_neighbor['fraudRatioAmongReceivers'] = (
        sender_neighbor['num_fraud_receivers'] / sender_neighbor['num_receivers'].replace(0, np.nan)
    )

    receiver_neighbor = df.groupby('nameDest').agg({
        'nameOrig': 'nunique',
        'sender_has_fraud': 'sum',
    }).reset_index()
    receiver_neighbor.columns = ['nameDest', 'num_senders', 'num_fraud_senders']
    receiver_neighbor['fraudRatioAmongSenders'] = (
        receiver_neighbor['num_fraud_senders'] / receiver_neighbor['num_senders'].replace(0, np.nan)
    )

    df = df.merge(
        sender_neighbor[['nameOrig', 'fraudRatioAmongReceivers']], on='nameOrig', how='left'
    )
    df = df.merge(
        receiver_neighbor[['nameDest', 'fraudRatioAmongSenders']], on='nameDest', how='left'
    )

    print(f"{split_name}: Sender aggregation preview")
    print(sender_agg.head())
    print(f"{split_name}: Receiver aggregation preview")
    print(receiver_agg.head())
    return df


def plot_sender_features(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Visualise sender-level aggregation features."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    hist_metrics = [
        ('totalSent', 'Total Sent'),
        ('meanSent', 'Mean Sent'),
        ('stdSent', 'Std Sent'),
        ('numSent', 'Transactions Sent'),
        ('fraudRatioAmongReceivers', 'Fraud Ratio Among Receivers'),
    ]
    for ax, (metric, title) in zip(axes.flatten(), hist_metrics + [('totalSent', 'Total Sent Boxplot')]):
        if title.endswith('Boxplot'):
            ax.boxplot(
                [
                    df[df['isFraud'] == 0]['totalSent'].dropna(),
                    df[df['isFraud'] == 1]['totalSent'].dropna(),
                ],
                labels=['Non-Fraud', 'Fraud'],
            )
            ax.set_title(f'{split_name}: {title}')
            ax.set_yscale('log')
            continue
        ax.hist(
            df[df['isFraud'] == 0][metric].dropna(),
            bins=50,
            alpha=0.6,
            label='Non-Fraud',
            color='blue',
            density=True,
        )
        ax.hist(
            df[df['isFraud'] == 1][metric].dropna(),
            bins=50,
            alpha=0.6,
            label='Fraud',
            color='red',
            density=True,
        )
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.set_title(f'{split_name}: {title}')
        if metric in {'totalSent', 'meanSent', 'stdSent'}:
            ax.set_xscale('log')
            ax.set_yscale('log')
        if metric == 'numSent':
            ax.set_yscale('log')
        ax.legend()

    plt.tight_layout()
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_sender_features.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    return df


def plot_receiver_features(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Visualise receiver-level aggregation features."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    hist_metrics = [
        ('totalReceived', 'Total Received'),
        ('meanReceived', 'Mean Received'),
        ('stdReceived', 'Std Received'),
        ('numReceived', 'Transactions Received'),
        ('fraudRatioAmongSenders', 'Fraud Ratio Among Senders'),
    ]
    for ax, (metric, title) in zip(axes.flatten(), hist_metrics + [('totalReceived', 'Total Received Boxplot')]):
        if title.endswith('Boxplot'):
            ax.boxplot(
                [
                    df[df['isFraud'] == 0]['totalReceived'].dropna(),
                    df[df['isFraud'] == 1]['totalReceived'].dropna(),
                ],
                labels=['Non-Fraud', 'Fraud'],
            )
            ax.set_title(f'{split_name}: {title}')
            ax.set_yscale('log')
            continue
        ax.hist(
            df[df['isFraud'] == 0][metric].dropna(),
            bins=50,
            alpha=0.6,
            label='Non-Fraud',
            color='blue',
            density=True,
        )
        ax.hist(
            df[df['isFraud'] == 1][metric].dropna(),
            bins=50,
            alpha=0.6,
            label='Fraud',
            color='red',
            density=True,
        )
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.set_title(f'{split_name}: {title}')
        if metric in {'totalReceived', 'meanReceived', 'stdReceived'}:
            ax.set_xscale('log')
            ax.set_yscale('log')
        if metric == 'numReceived':
            ax.set_yscale('log')
        ax.legend()

    plt.tight_layout()
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_receiver_features.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    return df


def plot_transaction_type_fractions(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Plot the distribution of transaction-type fractions per sender."""

    fraction_cols = [
        col for col in df.columns if col.startswith('fraction_') and not col.endswith('_recv')
    ]
    if not fraction_cols:
        print(f"No transaction type fraction columns found in {split_name}.")
        return df

    fig, axes = plt.subplots(1, len(fraction_cols), figsize=(6 * len(fraction_cols), 5))
    if len(fraction_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, fraction_cols):
        ax.hist(
            df[df['isFraud'] == 0][col].dropna(),
            bins=30,
            alpha=0.6,
            label='Non-Fraud',
            color='blue',
            density=True,
        )
        ax.hist(
            df[df['isFraud'] == 1][col].dropna(),
            bins=30,
            alpha=0.6,
            label='Fraud',
            color='red',
            density=True,
        )
        ax.set_xlabel(f'Fraction of {col.replace("fraction_", "")}', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{split_name}: Fraction of {col.replace("fraction_", "")}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_transaction_type_fractions.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    return df


def add_temporal_features(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Create hour/day categorical features and the related plots."""

    df['hourOfDay'] = df['step'] % 24
    df['dayOfWeek'] = (df['step'] // 24) % 7
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df['dayOfWeekName'] = df['dayOfWeek'].map({i: day_labels[i] for i in range(7)})
    print(f"{split_name}: Hour/day feature sample")
    display(df[['step', 'hourOfDay', 'dayOfWeek', 'dayOfWeekName']].head(10))

    fraud_rate_hour = df.groupby('hourOfDay')['isFraud'].mean().reset_index()
    fig1 = plt.figure(figsize=(10, 6))
    sns.lineplot(data=fraud_rate_hour, x='hourOfDay', y='isFraud', marker='o')
    plt.title(f'{split_name}: Fraud Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Rate')
    plt.grid(True, alpha=0.3)
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_fraud_rate_by_hour.png")
        fig1.savefig(plot_path)
        plt.close(fig1)
    else:
        plt.show()

    fraud_rate_day = df.groupby('dayOfWeekName')['isFraud'].mean().reset_index()
    day_order = day_labels
    fraud_rate_day['dayOfWeekName'] = pd.Categorical(
        fraud_rate_day['dayOfWeekName'], categories=day_order, ordered=True
    )
    fraud_rate_day = fraud_rate_day.sort_values('dayOfWeekName')
    fig2 = plt.figure(figsize=(8, 5))
    sns.barplot(
        data=fraud_rate_day,
        x='dayOfWeekName',
        y='isFraud',
        palette='magma',
        order=day_order,
    )
    plt.title(f'{split_name}: Fraud Rate by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Fraud Rate')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_fraud_rate_by_day.png")
        fig2.savefig(plot_path)
        plt.close(fig2)
    else:
        plt.show()

    fraud_pivot = df.pivot_table(
        index='dayOfWeekName',
        columns='hourOfDay',
        values='isFraud',
        aggfunc='mean',
    )
    fig3 = plt.figure(figsize=(12, 6))
    sns.heatmap(fraud_pivot, cmap='coolwarm', annot=False)
    plt.title(f'{split_name}: Fraud Rate Heatmap by Day and Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_fraud_rate_heatmap.png")
        fig3.savefig(plot_path)
        plt.close(fig3)
    else:
        plt.show()
        
    return df


def analyze_overlap_accounts(df: pd.DataFrame, split_name: str, **kwargs) -> pd.DataFrame:
    """Inspect accounts that appear both as senders and receivers."""

    senders = set(df['nameOrig'].unique())
    receivers = set(df['nameDest'].unique())
    overlap_accounts = senders.intersection(receivers)
    print(f"{split_name}: Total unique senders: {len(senders)}")
    print(f"{split_name}: Total unique receivers: {len(receivers)}")
    print(f"{split_name}: Accounts with dual roles: {len(overlap_accounts)}")

    suspicious_df = df[
        df['nameOrig'].isin(overlap_accounts) | df['nameDest'].isin(overlap_accounts)
    ]
    total_txn = df.shape[0]
    overlap_txn = suspicious_df.shape[0]
    overlap_pct = (overlap_txn / total_txn * 100) if total_txn else 0
    print(f"{split_name}: % of transactions involving dual-role accounts: {overlap_pct:.2f}%")

    fraud_overlap = suspicious_df['isFraud'].sum()
    fraud_rate_overlap = fraud_overlap / overlap_txn if overlap_txn else 0
    overall_fraud_rate = df['isFraud'].mean() if len(df) else 0
    print(f"{split_name}: Fraud rate (overlap accounts): {fraud_rate_overlap:.4f}")
    print(f"{split_name}: Overall fraud rate: {overall_fraud_rate:.4f}")
    print(f"{split_name}: Transaction type distribution for overlapping accounts")
    print(suspicious_df['type'].value_counts())
    return df


def add_unique_partner_counts(df: pd.DataFrame, split_name: str, **kwargs) -> pd.DataFrame:
    """Create numUniqueDest and numUniqueOrig features."""

    num_unique_dest = (
        df.groupby('nameOrig')['nameDest']
        .nunique()
        .reset_index(name='numUniqueDest')
    )
    num_unique_orig = (
        df.groupby('nameDest')['nameOrig']
        .nunique()
        .reset_index(name='numUniqueOrig')
    )
    df = df.merge(num_unique_dest, on='nameOrig', how='left')
    df = df.merge(num_unique_orig, on='nameDest', how='left')
    df[['numUniqueDest', 'numUniqueOrig']] = df[
        ['numUniqueDest', 'numUniqueOrig']
    ].fillna(0)
    return df


def add_transaction_recency(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Compute the hours since the previous transaction for each sender."""

    df = df.sort_values(by=['nameOrig', 'step']).reset_index(drop=True)
    df['transactionRecency'] = df.groupby('nameOrig')['step'].diff()
    df['transactionRecency'] = df['transactionRecency'].fillna(df['step'])

    fig = plt.figure(figsize=(8, 5))
    sns.kdeplot(df[df['isFraud'] == 0]['transactionRecency'], label='Non-Fraud', fill=True)
    sns.kdeplot(df[df['isFraud'] == 1]['transactionRecency'], label='Fraud', fill=True)
    plt.title(f'{split_name}: Transaction Recency Distribution by Fraud Label')
    plt.xlabel('Steps Since Last Transaction')
    plt.legend()
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_transaction_recency.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    return df


def create_transaction_sequence_features_efficient(
    df: pd.DataFrame, n_last_transactions: int
) -> pd.DataFrame:
    """Reproduce the sequence-based features used in the notebook."""

    df = df.sort_values(['nameOrig', 'step']).reset_index(drop=True)
    for i in range(1, n_last_transactions + 1):
        df[f'prev_type_{i}'] = df.groupby('nameOrig')['type'].shift(i)

    sequence_cols = [f'prev_type_{i}' for i in range(n_last_transactions, 0, -1)] + ['type']
    df['transaction_sequence'] = df[sequence_cols].apply(
        lambda x: '→'.join(
            [str(t) if pd.notna(t) else 'START' for t in x]
        ),
        axis=1,
    )

    sequence_freq = df['transaction_sequence'].value_counts(normalize=True).to_dict()
    df['sequence_frequency'] = df['transaction_sequence'].map(sequence_freq)

    df['is_cashin_transfer_cashout'] = (
        (df['prev_type_2'] == 'CASH_IN')
        & (df['prev_type_1'] == 'TRANSFER')
        & (df['type'] == 'CASH_OUT')
    ).astype(int)
    df['is_transfer_cashout'] = (
        (df['prev_type_1'] == 'TRANSFER') & (df['type'] == 'CASH_OUT')
    ).astype(int)
    df['is_cashin_transfer'] = (
        (df['prev_type_1'] == 'CASH_IN') & (df['type'] == 'TRANSFER')
    ).astype(int)
    df['is_cashout_transfer'] = (
        (df['prev_type_1'] == 'CASH_OUT') & (df['type'] == 'TRANSFER')
    ).astype(int)
    df['is_transfer_transfer'] = (
        (df['prev_type_1'] == 'TRANSFER') & (df['type'] == 'TRANSFER')
    ).astype(int)
    df['is_first_transfer'] = (
        (df['prev_type_1'] == 'START') & (df['type'] == 'TRANSFER')
    ).astype(int)
    df['is_cashin_cashout'] = (
        (df['prev_type_1'] == 'CASH_IN') & (df['type'] == 'CASH_OUT')
    ).astype(int)
    df['is_early_transaction'] = (
        (df['prev_type_2'] == 'START') | (df['prev_type_3'] == 'START')
    ).astype(int)

    from sklearn.preprocessing import LabelEncoder

    for i in range(1, n_last_transactions + 1):
        col_name = f'prev_type_{i}'
        encoder = LabelEncoder()
        df[f'{col_name}_encoded'] = encoder.fit_transform(df[col_name].fillna('NONE'))

    df['sequence_count'] = df.groupby(['nameOrig', 'transaction_sequence']).cumcount() + 1
    return df


def add_transaction_sequence_features(
    df: pd.DataFrame,
    split_name: str,
    n_last_transactions: int = 5,
    plots_dir: str | None = None,
) -> pd.DataFrame:
    """Add sequence-derived features and print descriptive stats."""

    df = create_transaction_sequence_features_efficient(df, n_last_transactions)

    fraud_patterns = (
        df[df['isFraud'] == 1]
        .groupby('transaction_sequence')
        .size()
        .sort_values(ascending=False)
    )
    legit_patterns = (
        df[df['isFraud'] == 0]
        .groupby('transaction_sequence')
        .size()
        .sort_values(ascending=False)
    )
    print(f"{split_name}: Top fraud sequences")
    print(fraud_patterns.head(10))
    print(f"{split_name}: Top legitimate sequences")
    print(legit_patterns.head(10))

    sequence_fraud_rate = df.groupby('transaction_sequence').agg({
        'isFraud': ['sum', 'count', 'mean'],
    }).round(4)
    sequence_fraud_rate.columns = ['fraud_count', 'total_count', 'fraud_rate']
    sequence_fraud_rate = sequence_fraud_rate.sort_values('fraud_rate', ascending=False)
    print(f"{split_name}: Sequences with highest fraud rates")
    print(sequence_fraud_rate.head(15))

    pattern_comparison = df.groupby('isFraud')[
        ['is_transfer_cashout', 'is_cashin_transfer_cashout', 'is_cashin_transfer']
    ].mean()
    fig = pattern_comparison.plot(kind='bar', figsize=(10, 6)).get_figure()
    plt.title(f'{split_name}: Fraud Pattern Occurrence')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.legend(title='Pattern Type')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_fraud_pattern_occurrence.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    print(pattern_comparison)

    fraud_sequences = df[df['is_transfer_cashout'] == 1].copy()
    fraud_sequences['time_since_prev'] = fraud_sequences.groupby('nameOrig')['step'].diff()
    print(f"{split_name}: Time between TRANSFER and CASH_OUT")
    print(fraud_sequences['time_since_prev'].describe())

    legit_sequences = df[(df['is_transfer_cashout'] == 0) & (df['type'] == 'CASH_OUT')].copy()
    legit_sequences['time_since_prev'] = legit_sequences.groupby('nameOrig')['step'].diff()
    print(f"{split_name}: Legitimate CASH_OUT timing")
    print(legit_sequences['time_since_prev'].describe())
    return df


def add_forwarding_features(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Measure % of funds forwarded within 24h."""

    incoming = df[['step', 'nameDest', 'amount']].copy()
    incoming.rename(columns={'nameDest': 'account', 'amount': 'amountReceived'}, inplace=True)
    outgoing = df[['step', 'nameOrig', 'amount']].copy()
    outgoing.rename(columns={'nameOrig': 'account', 'amount': 'amountSent'}, inplace=True)
    merged = incoming.merge(outgoing, on='account', how='inner')
    merged['hours_diff'] = merged['step_y'] - merged['step_x']
    within_24h = merged[(merged['hours_diff'] > 0) & (merged['hours_diff'] <= 24)]
    total_received = incoming.groupby('account')['amountReceived'].sum().reset_index()
    forwarded_24h = within_24h.groupby('account')['amountSent'].sum().reset_index(name='amountForwarded24h')
    funds_forwarding = total_received.merge(forwarded_24h, on='account', how='left')
    funds_forwarding['pctForwarded24h'] = (
        funds_forwarding['amountForwarded24h'] / funds_forwarding['amountReceived'] * 100
    )
    funds_forwarding['pctForwarded24h'] = funds_forwarding['pctForwarded24h'].fillna(0)
    df = df.merge(
        funds_forwarding[['account', 'pctForwarded24h']],
        left_on='nameOrig',
        right_on='account',
        how='left',
    )
    df.drop(columns='account', inplace=True)

    fig = plt.figure(figsize=(8, 5))
    sns.histplot(funds_forwarding['pctForwarded24h'], bins=30, kde=True)
    plt.title(f'{split_name}: % of Received Funds Forwarded Within 24 Hours')
    plt.xlabel('Percentage')
    plt.ylabel('Number of Accounts')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_forwarding_features.png")
        fig.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        
    return df


def add_pair_frequency_features(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Add sender–receiver pair frequency and visualise it."""

    pair_frequency = (
        df.groupby(['nameOrig', 'nameDest']).size().reset_index(name='pairFrequency')
    )
    df = df.merge(pair_frequency, on=['nameOrig', 'nameDest'], how='left')

    fig1 = plt.figure(figsize=(8, 5))
    sns.histplot(df['pairFrequency'], bins=50, kde=True)
    plt.title(f'{split_name}: Distribution of Sender–Receiver Pair Frequency')
    plt.xlabel('Number of Repeated Transactions')
    plt.ylabel('Count')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_pair_frequency_distribution.png")
        fig1.savefig(plot_path)
        plt.close(fig1)
    else:
        plt.show()

    fig2 = plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='isFraud', y='pairFrequency')
    plt.title(f'{split_name}: Pair Frequency by Fraud Label')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_pair_frequency_boxplot.png")
        fig2.savefig(plot_path)
        plt.close(fig2)
    else:
        plt.show()
        
    return df


def add_unique_partner_percentages(df: pd.DataFrame, split_name: str, plots_dir: str | None = None) -> pd.DataFrame:
    """Create pctUniqueDest and pctUniqueOrig percentage features and plots."""

    unique_dest = df.groupby('nameOrig')['nameDest'].nunique().reset_index(name='numUniqueDest_pct')
    total_sent = df.groupby('nameOrig').size().reset_index(name='totalSent_pct')
    sender_stats = unique_dest.merge(total_sent, on='nameOrig')
    sender_stats['pctUniqueDest'] = (
        sender_stats['numUniqueDest_pct'] / sender_stats['totalSent_pct'].replace(0, np.nan) * 100
    )

    unique_orig = df.groupby('nameDest')['nameOrig'].nunique().reset_index(name='numUniqueOrig_pct')
    total_received = df.groupby('nameDest').size().reset_index(name='totalReceived_pct')
    receiver_stats = unique_orig.merge(total_received, on='nameDest')
    receiver_stats['pctUniqueOrig'] = (
        receiver_stats['numUniqueOrig_pct'] / receiver_stats['totalReceived_pct'].replace(0, np.nan) * 100
    )

    df = df.merge(sender_stats[['nameOrig', 'pctUniqueDest']], on='nameOrig', how='left')
    df = df.merge(receiver_stats[['nameDest', 'pctUniqueOrig']], on='nameDest', how='left')
    df[['pctUniqueDest', 'pctUniqueOrig']] = df[
        ['pctUniqueDest', 'pctUniqueOrig']
    ].fillna(0)

    fig1 = plt.figure(figsize=(8, 5))
    sns.histplot(sender_stats['pctUniqueDest'].dropna(), bins=40, kde=True)
    plt.title(f'{split_name}: % of Unique Receivers per Sender')
    plt.xlabel('pctUniqueDest')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_pct_unique_receivers.png")
        fig1.savefig(plot_path)
        plt.close(fig1)
    else:
        plt.show()

    fig2 = plt.figure(figsize=(8, 5))
    sns.histplot(receiver_stats['pctUniqueOrig'].dropna(), bins=40, kde=True)
    plt.title(f'{split_name}: % of Unique Senders per Receiver')
    plt.xlabel('pctUniqueOrig')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_pct_unique_senders.png")
        fig2.savefig(plot_path)
        plt.close(fig2)
    else:
        plt.show()

    fig3 = plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='isFraud', y='pctUniqueDest')
    plt.title(f'{split_name}: % of Unique Receivers by Fraud Label')
    
    if plots_dir:
        plot_path = os.path.join(plots_dir, f"{split_name}_pct_unique_receivers_boxplot.png")
        fig3.savefig(plot_path)
        plt.close(fig3)
    else:
        plt.show()
        
    return df


def export_featured_datasets(
    split_frames: SplitMap,
    with_merchants_dir: str = "./data/raw(withMerchants)",
    without_merchants_dir: str = "./data/raw(withoutMerchants)",
) -> None:
    """Persist the engineered datasets with and without merchant destinations."""

    os.makedirs(with_merchants_dir, exist_ok=True)
    os.makedirs(without_merchants_dir, exist_ok=True)

    for split_name, df in split_frames.items():
        with_path = os.path.join(
            with_merchants_dir, f'FE_{split_name.lower()}_with_merchants.csv'
        )
        df.to_csv(with_path, index=False)
        print(f"Saved {split_name} (with merchants) to {with_path} | shape={df.shape}")

        merchants_only = df[df['nameDest'].str.contains('M')]
        if not merchants_only.empty:
            print(
                f"{split_name}: merchants isFraud unique: {merchants_only['isFraud'].unique()}"
            )
            if 'isFlaggedFraud' in merchants_only.columns:
                print(
                    f"{split_name}: merchants isFlaggedFraud unique: "
                    f"{merchants_only['isFlaggedFraud'].unique()}"
                )

        merchants_orig = df[df['nameOrig'].str.contains('M')]
        print(f"{split_name}: merchants as origin shape {merchants_orig.shape}")

        filtered_df = df[~df['nameDest'].str.contains('M')]
        
        print(f"\\n--- {split_name}: NON-Merchant Stats ---")
        if not filtered_df.empty:
            print(
                f"{split_name}: NON-merchants isFraud unique: {filtered_df['isFraud'].unique()}"
            )
            if 'isFlaggedFraud' in filtered_df.columns:
                print(
                    f"{split_name}: NON-merchants isFlaggedFraud unique: "
                    f"{filtered_df['isFlaggedFraud'].unique()}"
                )
        print(f"{split_name}: NON-merchants as destination shape {filtered_df.shape}")

        without_path = os.path.join(
            without_merchants_dir, f'FE_{split_name.lower()}_without_merchants.csv'
        )
        filtered_df.to_csv(without_path, index=False)
        print(
            f"Saved {split_name} (without merchants) to {without_path} | shape={filtered_df.shape}"
        )


def drop_unusable_columns(df: pd.DataFrame, split_name: str, **kwargs) -> pd.DataFrame:
    """Drop columns that are not usable for modeling as per Kaggle rules."""
    cols_to_drop = ['oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'oldbalanceDest']
    # Check which columns exist before trying to drop
    cols_exist = [col for col in cols_to_drop if col in df.columns]
    if cols_exist:
        df = df.drop(columns=cols_exist)
        print(f"Dropped columns from {split_name}: {cols_exist}")
    return df


def build_default_pipeline() -> List[FeatureStep]:
    """Return the ordered list of feature functions plus their kwargs."""

    return [
        (drop_unusable_columns, {}),
        (transaction_velocity, {}),
        (add_avg_amount_features, {}),
        (add_receiver_flow_features, {}),
        (plot_receiver_type_heatmap, {}),
        (add_transaction_type_features, {}),
        (add_sender_receiver_aggregations, {}),
        (plot_sender_features, {}),
        (plot_receiver_features, {}),
        (plot_transaction_type_fractions, {}),
        (add_temporal_features, {}),
        (analyze_overlap_accounts, {}),
        (add_unique_partner_counts, {}),
        (add_transaction_recency, {}),
        (add_transaction_sequence_features, {}),
        (add_forwarding_features, {}),
        (add_pair_frequency_features, {}),
        (add_unique_partner_percentages, {}),
    ]


def run_full_feature_pipeline(
    split_frames: SplitMap,
    pipeline: Iterable[FeatureStep] | None = None,
    *,
    export: bool = True,
    with_merchants_dir: str = "./data/FEwithMerchants",
    without_merchants_dir: str = "./data/FEwithoutMerchants",
    plots_dir: str | None = None,
) -> SplitMap:
    """Apply every feature step to each split and optionally export the outputs."""

    steps = list(pipeline) if pipeline is not None else build_default_pipeline()
    for func, kwargs in steps:
        if plots_dir:
            kwargs['plots_dir'] = plots_dir
        apply_to_splits(split_frames, func, **kwargs)
    if export:
        export_featured_datasets(split_frames, with_merchants_dir, without_merchants_dir)
    return split_frames


def main():
    """Load data from predefined paths, run feature pipeline, and export results."""
    
    # Define the paths to the data splits directly
    train_path = "./data/splits/train.csv"
    test_path = "./data/splits/test.csv"
    val_path = "./data/splits/val.csv"
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")
    print(f"Loading validation data from: {val_path}")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_val = pd.read_csv(val_path)

    split_frames = create_split_frames(df_train, df_test, df_val)
    
    print("\\nStarting feature engineering pipeline...")
    run_full_feature_pipeline(split_frames, export=True, plots_dir=plots_dir)
    print("Feature engineering pipeline complete.")

    print("\n--- Final Columns (with merchants) ---")
    for split_name, df in split_frames.items():
        print(f"\nColumns for {split_name} data:")
        print(df.columns)

    print("\n--- Final Columns (without merchants) ---")
    for split_name, df in split_frames.items():
        filtered_df = df[~df['nameDest'].str.contains('M')]
        print(f"\nColumns for {split_name} data (without merchants):")
        print(filtered_df.columns)

    

if __name__ == "__main__":
    main()


__all__ = [
    'create_split_frames',
    'apply_to_splits',
    'drop_unusable_columns',
    'transaction_velocity',
    'add_avg_amount_features',
    'add_receiver_flow_features',
    'plot_receiver_type_heatmap',
    'add_transaction_type_features',
    'add_sender_receiver_aggregations',
    'plot_sender_features',
    'plot_receiver_features',
    'plot_transaction_type_fractions',
    'add_temporal_features',
    'analyze_overlap_accounts',
    'add_unique_partner_counts',
    'add_transaction_recency',
    'add_transaction_sequence_features',
    'add_forwarding_features',
    'add_pair_frequency_features',
    'add_unique_partner_percentages',
    'export_featured_datasets',
    'build_default_pipeline',
    'run_full_feature_pipeline',
    'drop_unusable_columns',
]
