#!/bin/bash

# run_notebooks.sh
# Script to execute all notebooks in the correct order for the fraud detection pipeline
# This will run all analysis notebooks sequentially

set -e  # Exit on error

echo "=========================================="
echo "Fraud Detection Pipeline - Notebook Runner"
echo "=========================================="
echo ""

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "ERROR: Jupyter is not installed. Please install it first:"
    echo "  pip install jupyter"
    exit 1
fi

# Navigate to notebooks directory
cd "$(dirname "$0")/notebooks"

echo "Starting notebook execution pipeline..."
echo ""

# Phase 1: Data Preparation
echo "Phase 1: Data Preparation"
echo "-------------------------------------------"

echo "[1/7] Running EDA (0_EDA.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace 0_EDA.ipynb
echo "✓ EDA complete"
echo ""

echo "[2/7] Running Train/Test/Val Split (1a_train_test_val_split.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace 1a_train_test_val_split.ipynb
echo "✓ Data splitting complete"
echo ""

echo "[3/7] Running Resampling (1b_Resampling.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace 1b_Resampling.ipynb
echo "✓ Resampling complete"
echo ""

# Phase 2: Feature Engineering
echo "Phase 2: Feature Engineering"
echo "-------------------------------------------"

echo "[4/7] Running Feature Engineering Pipeline (2_Feature_Engineering_Pipeline.py)..."
python 2_Feature_Engineering_Pipeline.py
echo "✓ Feature engineering complete"
echo ""

# Phase 3: Model Training
echo "Phase 3: Model Training"
echo "-------------------------------------------"

echo "[5/7] Running Logistic Regression (3_Logistic_Regression.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace 3_Logistic_Regression.ipynb
echo "✓ Logistic Regression complete"
echo ""

echo "[6/7] Running Neural Network (4_Neural_Network.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace 4_Neural_Network.ipynb
echo "✓ Neural Network complete"
echo ""

echo "[7/7] Running XGBoost (5_XGBoost.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace 5_XGBoost.ipynb
echo "✓ XGBoost complete"
echo ""

echo "[8/8] Running Ensemble Model (6_Ensemble_Model.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace 6_Ensemble_Model.ipynb
echo "✓ Ensemble Model complete"
echo ""

# Done
echo "=========================================="
echo "Pipeline execution complete!"
echo "=========================================="
echo ""
echo "You can now run the Streamlit app:"
echo "  streamlit run src/app.py"
echo ""
