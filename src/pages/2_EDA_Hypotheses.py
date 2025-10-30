import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import os
import subprocess
import json
import shutil
import signal
import os as _os
from pathlib import Path
import time

st.set_page_config(
    page_title="EDA: Fraud Hypotheses",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 EDA — Hypotheses Checks")

st.markdown(
    """
    This page visualises the key hypotheses from the exploratory data analysis.

    """
)

# --- Dataset paths (match other pages) ---
DATASET_PATHS = {
    # Raw dataset (recommended for EDA)
    "Raw synthetic dataset": "./data/raw/financial-fraud-detection-dataset/Synthetic_Financial_datasets_log.csv",
    "Test without Merchants": "./data/FEwithoutMerchants/FE_test_without_merchants.csv",
    "Train without Merchants": "./data/FEwithoutMerchants/FE_train_without_merchants.csv",
    "Validation without Merchants": "./data/FEwithoutMerchants/FE_validation_without_merchants.csv",
    "Test with Merchants": "./data/FEwithMerchants/FE_test_with_merchants.csv",
    "Train with Merchants": "./data/FEwithMerchants/FE_train_with_merchants.csv",
    "Validation with Merchants": "./data/FEwithMerchants/FE_validation_with_merchants.csv",
}


@st.cache_data
def load_data(path: str, nrows: int | None = None):
    try:
        return pd.read_csv(path, nrows=nrows)
    except FileNotFoundError:
        st.error(f"Could not find `{path}`. Run the feature-engineering pipeline first or pick another dataset.")
        return None
    except Exception as e:
        st.error(f"Error loading `{path}`: {e}")
        return None


st.sidebar.header("Dataset")
dataset_choice = st.sidebar.selectbox("Choose dataset:", list(DATASET_PATHS.keys()))

# Allow sampling for very large raw files
st.sidebar.markdown("**Load options**")
rows_to_load = st.sidebar.number_input("Rows to load (0 = all)", min_value=0, value=100000, step=1000)
nrows = None if int(rows_to_load) == 0 else int(rows_to_load)

# Precomputed artifacts directory and raw path
PROCESSED_DIR = Path("data/processed/eda")
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
RAW_DEFAULT = "./data/raw/financial-fraud-detection-dataset/Synthetic_Financial_datasets_log.csv"

# Toggle: prefer precomputed artifacts when present
use_cache_default = MANIFEST_PATH.exists()
use_cache = st.sidebar.checkbox('Use precomputed artifacts when available', value=use_cache_default)

df = load_data(DATASET_PATHS[dataset_choice], nrows=nrows)
if nrows is not None:
    st.sidebar.info(f"Loading first {nrows:,} rows (sampling).")


def run_precompute(raw_path: str = RAW_DEFAULT, chunksize: int = 200_000, max_amount_samples: int = 200_000):
    """Run the precompute script and wait for it to finish. Returns (success, output)."""
    cmd = ["python3", "scripts/precompute_eda.py", "--input", raw_path, "--chunksize", str(chunksize), "--max-amount-samples", str(max_amount_samples)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, proc.stdout + proc.stderr
    except subprocess.CalledProcessError as e:
        return False, (e.stdout or '') + (e.stderr or '')


def start_precompute_bg(raw_path: str = RAW_DEFAULT, chunksize: int = 200_000, max_amount_samples: int = 200_000):
    """Start precompute as a background process and write logs to data/processed/eda/precompute.log.
    Stores pid in st.session_state['precompute_pid'].
    """
    ensure_dir = PROCESSED_DIR
    ensure_dir.mkdir(parents=True, exist_ok=True)
    log_path = ensure_dir / 'precompute.log'
    cmd = ["python3", "scripts/precompute_eda.py", "--input", raw_path, "--chunksize", str(chunksize), "--max-amount-samples", str(max_amount_samples), "--outdir", str(PROCESSED_DIR)]
    # Open log file and start subprocess
    logfile = open(log_path, 'a', encoding='utf-8')
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=logfile, text=True)
    st.session_state['precompute_pid'] = proc.pid
    st.session_state['precompute_log'] = str(log_path)
    st.session_state['precompute_start'] = time.time()
    return proc.pid


def stop_precompute_bg():
    pid = st.session_state.get('precompute_pid')
    if not pid:
        return False, 'No running precompute found in session.'
    try:
        _os.kill(pid, signal.SIGTERM)
        # cleanup session state
        st.session_state.pop('precompute_pid', None)
        return True, f'Process {pid} terminated.'
    except Exception as e:
        return False, str(e)


def read_log_tail(path: Path, max_lines: int = 1000):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # return last max_lines
        return ''.join(lines[-max_lines:])
    except Exception:
        return ''


def load_plotly_json(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            j = f.read()
        fig = pio.from_json(j)
        return fig
    except Exception:
        return None


if df is None:
    st.stop()

st.info(f"Loaded **{len(df):,}** rows from **{dataset_choice}**")

# Helper: check for columns and show info
def has_cols(*cols):
    missing = [c for c in cols if c not in df.columns]
    return len(missing) == 0, missing


# -----------------------
# Precompute controls & load cached artifacts
# -----------------------
recompute_clicked = st.sidebar.button("Recompute precomputed artifacts (slow)")
if recompute_clicked:
    # Start the precompute in background and show logs with refresh control
    raw_path_input = DATASET_PATHS.get("Raw synthetic dataset", RAW_DEFAULT)
    st.sidebar.info(f"Starting background precompute on: {raw_path_input}")
    pid = start_precompute_bg(raw_path=raw_path_input)
    st.sidebar.success(f"Precompute started (pid={pid}). Use the 'Refresh logs' button to view progress.")

# Controls to manage background precompute
if 'precompute_pid' in st.session_state:
    pid = st.session_state['precompute_pid']
    st.sidebar.markdown(f"**Precompute running (pid={pid})**")
    if st.sidebar.button('Stop precompute'):
        ok, msg = stop_precompute_bg()
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

    # Log viewer
    log_path = Path(st.session_state.get('precompute_log', PROCESSED_DIR / 'precompute.log'))
    if log_path.exists():
        if st.sidebar.button('Refresh logs'):
            pass
        logs = read_log_tail(log_path)
        st.sidebar.text_area('Precompute logs (tail)', value=logs, height=300)
    else:
        st.sidebar.info('No precompute log yet. Refresh after starting the job.')


# Attempt to load precomputed plot artifacts if the user enabled the cache
skip_fraud_by_type = False
skip_pair_counts = False
if use_cache and MANIFEST_PATH.exists():
    plots_dir = PROCESSED_DIR / 'plots'
    if (plots_dir / 'fraud_by_type.json').exists():
        fig_pre = load_plotly_json(plots_dir / 'fraud_by_type.json')
        if fig_pre is not None:
            st.subheader("Fraud rate by transaction type (precomputed)")
            st.plotly_chart(fig_pre, use_container_width=True)
            skip_fraud_by_type = True

    if (plots_dir / 'pair_counts_hist.json').exists():
        fig_pair = load_plotly_json(plots_dir / 'pair_counts_hist.json')
        if fig_pair is not None:
            st.subheader("Transactions per sender–receiver pair (precomputed)")
            st.plotly_chart(fig_pair, use_container_width=True)
            skip_pair_counts = True


# -----------------------
# Delete processed artifacts (optional)
# -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Processed artifact housekeeping**")
confirm_delete = st.sidebar.checkbox('I understand this will delete processed artifacts', value=False)
if st.sidebar.button('Delete processed artifacts'):
    if not confirm_delete:
        st.sidebar.error('Please check the confirmation checkbox to delete artifacts.')
    else:
        if PROCESSED_DIR.exists():
            try:
                shutil.rmtree(PROCESSED_DIR)
                st.sidebar.success('Processed artifacts deleted.')
                # recreate directory so UI logic expecting it doesn't break
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                # clear manifest-related flags
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f'Failed to delete processed artifacts: {e}')
        else:
            st.sidebar.info('No processed artifacts directory found.')



### 1) Fraud concentrated in TRANSFER and CASH_OUT (Fraud rate by transaction type)
st.header("Fraud rate by transaction type")
if not skip_fraud_by_type:
    if has_cols('type', 'isFraud')[0]:
        fraud_by_type = df.groupby('type', dropna=False)['isFraud'].mean().reset_index()
        fraud_by_type['fraud_rate_pct'] = fraud_by_type['isFraud'] * 100
        fig = px.bar(fraud_by_type.sort_values('fraud_rate_pct', ascending=False),
                     x='type', y='fraud_rate_pct', text='fraud_rate_pct',
                     labels={'fraud_rate_pct': 'Fraud rate (%)', 'type': 'Transaction type'},
                     title='Fraud rate (%) by transaction type')
        st.plotly_chart(fig, use_container_width=True)
    else:
        _, missing = has_cols('type', 'isFraud')
        st.warning(f"Skipping fraud-by-type chart. Missing columns: {missing}")
else:
    st.info('Showing precomputed fraud-by-type plot above (cached).')


### 2) Transaction amounts: fraud vs non-fraud
st.header("Transaction amount distributions: Fraud vs Non-Fraud")
if has_cols('amount', 'isFraud')[0]:
    # use log scale for better visibility, but keep original values too
    plot_df = df[['amount', 'isFraud']].copy()
    plot_df['label'] = plot_df['isFraud'].map({0: 'Non-Fraud', 1: 'Fraud'})

    fig_box = px.box(plot_df, x='label', y='amount', log_y=True,
                     points='outliers', title='Amount distribution (log y) — Fraud vs Non-Fraud')
    st.plotly_chart(fig_box, use_container_width=True)

    # Fraud probability across amount ranges
    st.subheader('Fraud probability across transaction amount ranges')
    try:
        # create quantile bins to examine tail behaviour
        bins = np.quantile(plot_df['amount'].clip(lower=1), np.linspace(0, 1, 11))
        plot_df['amount_bin'] = pd.cut(plot_df['amount'].clip(lower=1), bins=bins, include_lowest=True)
        prob = plot_df.groupby('amount_bin')['isFraud'].mean().reset_index()
        prob['bin_label'] = prob['amount_bin'].astype(str)
        prob['fraud_pct'] = prob['isFraud'] * 100
        fig_prob = px.bar(prob, x='bin_label', y='fraud_pct', title='Fraud % by amount decile', labels={'fraud_pct': 'Fraud rate (%)', 'bin_label': 'Amount decile'})
        st.plotly_chart(fig_prob, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute fraud-by-amount-bins: {e}")
else:
    _, missing = has_cols('amount', 'isFraud')
    st.warning(f"Skipping amount distribution charts. Missing columns: {missing}")


### 3) High hourly averages correlated with fraud, esp TRANSFER & CASH_OUT
st.header("Hourly average amounts and fraud (per-sender)")
hour_col = None
if 'hour' in df.columns:
    hour_col = 'hour'
elif 'step' in df.columns:
    # common in synthetic dataset: step is hours since start
    hour_col = 'step'

if has_cols('nameOrig', 'amount', 'isFraud', 'type')[0] and hour_col is not None:
    # derive hour-of-day if step present (mod 24)
    local = df[['nameOrig', 'amount', 'isFraud', 'type', hour_col]].copy()
    if hour_col == 'step':
        local['hour_of_day'] = (local['step'] % 24).astype(int)
    else:
        local['hour_of_day'] = local[hour_col].astype(int)

    # compute sender-hour mean
    sender_hour = local.groupby(['nameOrig', 'hour_of_day'])['amount'].mean().reset_index(name='avg_amount')
    # mark whether sender has any fraud
    sender_fraud_flag = df.groupby('nameOrig')['isFraud'].max().reset_index()
    sender_hour = sender_hour.merge(sender_fraud_flag, on='nameOrig', how='left')

    # plot distribution of avg hourly amounts for fraud vs non-fraud senders
    sender_hour['label'] = sender_hour['isFraud'].map({0: 'Non-Fraud Sender-hour', 1: 'Fraud Sender-hour'})
    fig = px.violin(sender_hour, x='label', y='avg_amount', log_y=True, points='outliers', title='Distribution of sender-hour average amounts (log y)')
    st.plotly_chart(fig, use_container_width=True)

    # focus on TRANSFER and CASH_OUT
    types_focus = ['TRANSFER', 'CASH_OUT']
    if 'type' in local.columns:
        local_focus = local[local['type'].isin(types_focus)].copy()
        # sender average amount across all hours for these types
        sender_focus_avg = local_focus.groupby('nameOrig')['amount'].mean().reset_index(name='mean_amount')
        sender_focus_avg = sender_focus_avg.merge(sender_fraud_flag, on='nameOrig', how='left')
        sender_focus_avg['label'] = sender_focus_avg['isFraud'].map({0: 'Non-Fraud Sender', 1: 'Fraud Sender'})
        fig2 = px.box(sender_focus_avg, x='label', y='mean_amount', log_y=True, title='Avg amount per sender for TRANSFER & CASH_OUT')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info('Transaction type not found; skipping TRANSFER/CASH_OUT specific chart.')
else:
    missing_cols = [c for c in ['nameOrig', 'amount', 'isFraud', 'type'] if c not in df.columns]
    st.warning(f"Skipping hourly-average analysis. Missing columns or hour/step not present. Missing: {missing_cols}")


### 4) Fraudulent users often execute many transactions in short time spans (test)
st.header("Transaction burstiness per sender")
if has_cols('nameOrig', 'step', 'isFraud')[0] or has_cols('nameOrig', 'timestamp', 'isFraud')[0]:
    # prefer step, otherwise timestamp if present
    if 'step' in df.columns:
        time_col = 'step'
        time_series = df[['nameOrig', 'step', 'isFraud']].copy()
    else:
        time_col = 'timestamp'
        time_series = df[['nameOrig', 'timestamp', 'isFraud']].copy()

    # compute per-sender transaction counts in rolling 24-hour windows is heavy; instead compute inter-transaction time
    time_series = time_series.sort_values(['nameOrig', time_col])
    time_series['prev_time'] = time_series.groupby('nameOrig')[time_col].shift(1)
    time_series['delta'] = time_series[time_col] - time_series['prev_time']
    # per-sender median delta
    burst = time_series.groupby('nameOrig')['delta'].median().reset_index()
    burst = burst.merge(df.groupby('nameOrig')['isFraud'].max().reset_index(), on='nameOrig', how='left')
    burst['label'] = burst['isFraud'].map({0: 'Non-Fraud Sender', 1: 'Fraud Sender'})
    fig = px.box(burst, x='label', y='delta', title='Median inter-transaction time by sender (lower = bursty)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Observation: if fraudsters are bursty we expect fraud senders to have much lower median inter-transaction times.
    """)
else:
    st.warning('Skipping burstiness analysis — required time column not found (step or timestamp).')


### 5) Unique connections: how many unique receivers per sender and unique senders per receiver
st.header("Unique connections per account")
if has_cols('nameOrig', 'nameDest')[0]:
    uniq_out = df.groupby('nameOrig')['nameDest'].nunique().reset_index(name='unique_receivers')
    uniq_out = uniq_out.merge(df.groupby('nameOrig')['isFraud'].max().reset_index(), on='nameOrig', how='left')
    uniq_out['label'] = uniq_out['isFraud'].map({0: 'Non-Fraud Sender', 1: 'Fraud Sender'})
    fig = px.histogram(uniq_out, x='unique_receivers', color='label', barmode='overlay', nbins=50, title='Unique receivers per sender')
    st.plotly_chart(fig, use_container_width=True)

    uniq_in = df.groupby('nameDest')['nameOrig'].nunique().reset_index(name='unique_senders')
    uniq_in = uniq_in.merge(df.groupby('nameDest')['isFraud'].max().reset_index(), left_on='nameDest', right_on='nameDest', how='left')
    # For receivers we may not have isFraud defined the same way; show distribution
    fig2 = px.histogram(uniq_in, x='unique_senders', nbins=50, title='Unique senders per receiver')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning('Skipping unique-connection charts — missing nameOrig/nameDest columns.')


### 6) Pair frequency (how many times a sender–receiver pair transacts)
st.header("Sender–receiver pair frequency")
if not skip_pair_counts:
    if has_cols('nameOrig', 'nameDest', 'isFraud')[0]:
        pair_counts = df.groupby(['nameOrig', 'nameDest']).size().reset_index(name='count')
        # attach fraud flag: if any transaction between the pair is fraud
        pair_fraud = df.groupby(['nameOrig', 'nameDest'])['isFraud'].max().reset_index(name='isFraud_any')
        pair_counts = pair_counts.merge(pair_fraud, on=['nameOrig', 'nameDest'], how='left')
        # show distribution of counts
        fig = px.histogram(pair_counts, x='count', color=pair_counts['isFraud_any'].map({0: 'Non-Fraud Pair', 1: 'Fraud Pair'}), nbins=20, title='Distribution of transactions per sender–receiver pair')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('''
        Observation: a concentration at count=1 indicates most pairs transact only once ("hit-and-run").
        ''')
    else:
        st.warning('Skipping pair-frequency analysis — missing columns.')
else:
    st.info('Showing precomputed pair-frequency plot above (cached).')


### 7) Simple sequence check: transfer -> cash_out pattern
st.header('Simple transfer → cash_out sequence checks')
if has_cols('nameOrig', 'nameDest', 'type', 'isFraud', 'step')[0]:
    # look for sequences within same account where a TRANSFER is followed by CASH_OUT within next N steps (e.g., 1 step)
    seq_df = df[['nameOrig', 'nameDest', 'type', 'isFraud', 'step']].copy()
    seq_df = seq_df.sort_values(['nameOrig', 'step'])
    seq_df['next_type'] = seq_df.groupby('nameOrig')['type'].shift(-1)
    seq_df['next_step'] = seq_df.groupby('nameOrig')['step'].shift(-1)
    seq_df['time_diff'] = seq_df['next_step'] - seq_df['step']
    pattern = seq_df[(seq_df['type'] == 'TRANSFER') & (seq_df['next_type'] == 'CASH_OUT')]
    # fraction of transfer transactions that are immediately followed by cash_out (within 1 step)
    immediate = pattern[pattern['time_diff'] <= 1]
    pct_immediate = 100 * len(immediate) / max(1, len(seq_df[seq_df['type'] == 'TRANSFER']))
    st.write(f"{pct_immediate:.3f}% of TRANSFERs are followed by CASH_OUT within 1 step (per-sender sequencing).")
    st.write(f"Total matching sequences found: {len(pattern)}; immediate (<=1 step): {len(immediate)}")
    # show top examples
    st.subheader('Examples of TRANSFER → CASH_OUT sequences')
    st.dataframe(pattern.head(50))
else:
    st.warning('Skipping sequence check — one of required columns missing: nameOrig/nameDest/type/step/isFraud')


st.markdown('---')
st.write('Notes: charts are interactive. If a chart is missing, check the dataset selected and the column names in your FE CSV.')
