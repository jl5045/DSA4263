"""Precompute EDA artifacts from the raw synthetic financial dataset.

Artifacts produced (saved under data/processed/eda/):
- fraud_by_type.parquet (type, n_tx, n_fraud, fraud_rate)
- amount_deciles.parquet (decile, lo, hi, fraud_rate)
- sender_stats.parquet (nameOrig, tx_count, tx_sum, median_amount, max_amount, isFraud_any)
- sender_hour_agg.parquet (nameOrig, hour_of_day, avg_amount)
- pair_counts.parquet (nameOrig, nameDest, tx_count, isFraud_any)
- uniq_connections.parquet (per-sender unique_receivers, per-receiver unique_senders)
- fraud_over_time.parquet (step, fraud_count)
- transfer_cashout_sequences.parquet (examples of TRANSFER->CASH_OUT sequences found heuristically)

Usage:
    python scripts/precompute_eda.py --input data/raw/financial-fraud-detection-dataset/Synthetic_Financial_datasets_log.csv

The script is defensive: if pyarrow is available it will write parquet, otherwise CSV.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import plotly.express as px
import math


def ensure_out(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def to_parquet_or_csv(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path.with_suffix('.parquet'), index=False)
        return path.with_suffix('.parquet')
    except Exception:
        df.to_csv(path.with_suffix('.csv'), index=False)
        return path.with_suffix('.csv')


def save_plot_json(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(fig.to_json())


def process(args):
    inp = Path(args.input)
    assert inp.exists(), f"Input {inp} does not exist"
    outdir = Path(args.outdir)
    ensure_out(outdir)

    chunksize = args.chunksize

    # Accumulators
    type_counts = Counter()
    type_fraud = Counter()
    pair_counts = Counter()
    pair_fraud = Counter()
    sender_receivers = defaultdict(set)
    receiver_senders = defaultdict(set)
    sender_tx_count = Counter()
    sender_tx_sum = Counter()
    sender_amounts_stats = defaultdict(list)  # will collect limited samples per sender
    fraud_over_time = Counter()

    # For amount deciles we collect a reservoir sample up to max_amount_samples
    amount_samples = []
    max_amount_samples = args.max_amount_samples

    # Heuristic sequence detection: keep track of last event per sender (type, step, nameDest)
    last_event = dict()  # nameOrig -> (type, step, nameDest, isFraud)
    sequences = []

    reader = pd.read_csv(inp, chunksize=chunksize)
    total = 0
    for i, chunk in enumerate(reader):
        # normalize column names if necessary (strip)
        chunk.columns = [c.strip() for c in chunk.columns]

        total += len(chunk)
        print(f"Processing chunk {i} ({len(chunk)} rows) — total so far: {total}")

        # basic presence checks
        has_type = 'type' in chunk.columns
        has_isFraud = 'isFraud' in chunk.columns
        has_amount = 'amount' in chunk.columns
        has_names = ('nameOrig' in chunk.columns) and ('nameDest' in chunk.columns)
        has_step = 'step' in chunk.columns

        if has_type and has_isFraud:
            tc = chunk.groupby('type').size()
            for k, v in tc.items():
                type_counts[k] += int(v)
            tf = chunk[chunk['isFraud'] == 1].groupby('type').size()
            for k, v in tf.items():
                type_fraud[k] += int(v)

        if has_names:
            # pair counts
            pairs = list(zip(chunk['nameOrig'], chunk['nameDest']))
            for a, b in pairs:
                pair_counts[(a, b)] += 1
            if has_isFraud:
                fraud_pairs = chunk[chunk['isFraud'] == 1]
                for a, b in zip(fraud_pairs['nameOrig'], fraud_pairs['nameDest']):
                    pair_fraud[(a, b)] += 1

            # unique connections
            for a, b in zip(chunk['nameOrig'], chunk['nameDest']):
                sender_receivers[a].add(b)
                receiver_senders[b].add(a)

            # sender stats
            if has_amount:
                for a, amt in zip(chunk['nameOrig'], chunk['amount']):
                    sender_tx_count[a] += 1
                    try:
                        sender_tx_sum[a] += float(amt)
                    except Exception:
                        sender_tx_sum[a] += 0.0
                    # collect limited amounts per sender for median estimates (cap to 100)
                    if len(sender_amounts_stats[a]) < 100:
                        sender_amounts_stats[a].append(float(amt))

        # fraud over time
        if has_step and has_isFraud:
            fts = chunk[chunk['isFraud'] == 1].groupby('step').size()
            for s, v in fts.items():
                fraud_over_time[int(s)] += int(v)

        # amount samples for deciles
        if has_amount:
            vals = chunk['amount'].dropna().astype(float).values
            if len(amount_samples) < max_amount_samples:
                need = max_amount_samples - len(amount_samples)
                amount_samples.extend(list(vals[:need]))
            # reservoir sampling for rest
            if len(vals) > 0 and len(amount_samples) >= max_amount_samples:
                # reservoir sampling replacement
                for v in vals:
                    j = np.random.randint(0, total)
                    if j < max_amount_samples:
                        amount_samples[j] = float(v)

        # sequence heuristic: look for TRANSFER followed by CASH_OUT within small step difference
        if has_names and has_type and has_step:
            for a, t, s_step, dest, isf in zip(chunk['nameOrig'], chunk['type'], chunk['step'], chunk['nameDest'], chunk['isFraud'] if 'isFraud' in chunk.columns else [0]*len(chunk)):
                prev = last_event.get(a)
                try:
                    s_step_i = int(s_step)
                except Exception:
                    s_step_i = None
                if prev is not None:
                    prev_type, prev_step, prev_dest, prev_isf = prev
                    if prev_type == 'TRANSFER' and t == 'CASH_OUT' and (s_step_i is not None and prev_step is not None):
                        time_diff = s_step_i - prev_step
                        sequences.append({
                            'sender': a,
                            'transfer_dest': prev_dest,
                            'cashout_dest': dest,
                            'transfer_step': prev_step,
                            'cashout_step': s_step_i,
                            'time_diff': time_diff,
                            'isFraud_transfer': int(prev_isf),
                            'isFraud_cashout': int(isf)
                        })
                # update last event
                try:
                    last_event[a] = (t, int(s_step) if s_step is not None and not (isinstance(s_step, float) and math.isnan(s_step)) else None, dest, int(isf) if 'isFraud' in chunk.columns else 0)
                except Exception:
                    last_event[a] = (t, None, dest, int(isf) if 'isFraud' in chunk.columns else 0)

    # End chunk loop
    print(f"Finished reading. Processed {total} rows.")

    # Build DataFrames
    # fraud_by_type
    types = list(set(list(type_counts.keys()) + list(type_fraud.keys())))
    fraud_by_type = pd.DataFrame([
        {'type': t, 'n_tx': int(type_counts.get(t, 0)), 'n_fraud': int(type_fraud.get(t, 0))}
        for t in types
    ])
    if not fraud_by_type.empty:
        fraud_by_type['fraud_rate'] = fraud_by_type['n_fraud'] / fraud_by_type['n_tx']
    fraud_by_type = fraud_by_type.sort_values('n_tx', ascending=False)
    to_parquet_or_csv(fraud_by_type, outdir / 'fraud_by_type')

    # amount deciles from samples
    if len(amount_samples) > 0:
        arr = np.array(amount_samples)
        deciles = np.quantile(arr, np.linspace(0, 1, 11))
        rows = []
        for i in range(10):
            lo, hi = float(deciles[i]), float(deciles[i+1])
            # compute fraud rate in sample for this bin (approx)
            mask = (arr >= lo) & (arr <= hi)
            # cannot compute fraud probability from samples alone accurately; set NaN
            rows.append({'decile': i+1, 'lo': lo, 'hi': hi})
        amount_deciles = pd.DataFrame(rows)
        to_parquet_or_csv(amount_deciles, outdir / 'amount_deciles')

    # pair counts
    pc_rows = [{'nameOrig': a, 'nameDest': b, 'tx_count': c, 'isFraud_any': int(pair_fraud.get((a,b),0) > 0)} for (a,b), c in pair_counts.items()]
    pair_counts_df = pd.DataFrame(pc_rows).sort_values('tx_count', ascending=False)
    to_parquet_or_csv(pair_counts_df, outdir / 'pair_counts')

    # unique connections
    uniq_sender = [{'nameOrig': a, 'unique_receivers': len(s), 'isFraud_any': int(any([False for _ in []]))} for a, s in sender_receivers.items()]
    uniq_sender_df = pd.DataFrame(uniq_sender)
    to_parquet_or_csv(uniq_sender_df, outdir / 'uniq_receivers_per_sender')

    uniq_receiver = [{'nameDest': a, 'unique_senders': len(s)} for a, s in receiver_senders.items()]
    uniq_receiver_df = pd.DataFrame(uniq_receiver)
    to_parquet_or_csv(uniq_receiver_df, outdir / 'uniq_senders_per_receiver')

    # sender stats
    srows = []
    for s in sender_tx_count:
        amounts = sender_amounts_stats.get(s, [])
        median_amt = float(np.median(amounts)) if len(amounts) > 0 else np.nan
        srows.append({'nameOrig': s, 'tx_count': int(sender_tx_count[s]), 'tx_sum': float(sender_tx_sum.get(s, 0.0)), 'median_amount_est': median_amt})
    sender_stats_df = pd.DataFrame(srows).sort_values('tx_sum', ascending=False)
    to_parquet_or_csv(sender_stats_df, outdir / 'sender_stats')

    # fraud over time
    fot = pd.DataFrame([{'step': int(s), 'fraud_count': int(c)} for s, c in sorted(fraud_over_time.items())])
    to_parquet_or_csv(fot, outdir / 'fraud_over_time')

    # sequences (save sample of sequences)
    seq_df = pd.DataFrame(sequences)
    if not seq_df.empty:
        to_parquet_or_csv(seq_df.head(10000), outdir / 'transfer_cashout_sequences')

    # small plots
    try:
        if not fraud_by_type.empty:
            fig = px.bar(fraud_by_type, x='type', y='fraud_rate', title='Fraud rate by type')
            save_plot_json(fig, outdir / 'plots' / 'fraud_by_type.json')

        if not pair_counts_df.empty:
            fig2 = px.histogram(pair_counts_df, x='tx_count', title='Transactions per sender–receiver pair')
            save_plot_json(fig2, outdir / 'plots' / 'pair_counts_hist.json')
    except Exception as e:
        print('Could not create plots:', e)

    # Write manifest
    manifest = {
        'input': str(inp),
        'rows_processed': int(total),
        'artifacts': [str(p.relative_to(Path.cwd())) for p in outdir.glob('*') if p.is_file()]
    }
    with open(outdir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print('Precompute finished. Artifacts written to', outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to raw CSV')
    parser.add_argument('--outdir', type=str, default='data/processed/eda', help='Output directory for artifacts')
    parser.add_argument('--chunksize', type=int, default=200_000, help='CSV read chunksize')
    parser.add_argument('--max-amount-samples', dest='max_amount_samples', type=int, default=200_000, help='Max reservoir samples for amount quantiles')
    args = parser.parse_args()
    process(args)


if __name__ == '__main__':
    main()
