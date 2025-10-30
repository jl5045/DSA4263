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

echo "Precompute: input=${INPUT} outdir=${OUTDIR}"

python3 scripts/precompute_eda.py --input "$INPUT" --outdir "$OUTDIR"

echo "Precompute finished. Starting Streamlit..."

streamlit run src/app.py
