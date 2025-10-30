#!/usr/bin/env bash
# Run the precompute script (chunked, disk-backed) and then launch Streamlit.
# Usage:
#   ./scripts/run_precompute_and_streamlit.sh [INPUT_CSV] [OUTDIR] [CHUNKSIZE] [MAX_SAMPLES]
# Examples:
#   ./scripts/run_precompute_and_streamlit.sh
#   ./scripts/run_precompute_and_streamlit.sh data/raw/financial-fraud-detection-dataset/Synthetic_Financial_datasets_log.csv

set -euo pipefail

INPUT=${1:-data/raw/financial-fraud-detection-dataset/Synthetic_Financial_datasets_log.csv}
OUTDIR=${2:-data/processed/eda}
CHUNKSIZE=${3:-200000}
MAX_SAMPLES=${4:-200000}

echo "Precompute: input=${INPUT} outdir=${OUTDIR} chunksize=${CHUNKSIZE} max_samples=${MAX_SAMPLES}"

python3 scripts/precompute_eda.py --input "$INPUT" --outdir "$OUTDIR" --chunksize "$CHUNKSIZE" --max-amount-samples "$MAX_SAMPLES"

echo "Precompute finished. Starting Streamlit..."

streamlit run src/app.py
