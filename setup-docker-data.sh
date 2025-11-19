#!/bin/bash
# Script to prepare lightweight data directory for Docker deployment
# This copies only test, validation, and downsampled train data (excludes huge train files)

echo "Preparing Docker data directory..."

# Create directories
mkdir -p data-docker/FEwithoutMerchants
mkdir -p data-docker/FEwithMerchants

echo "Copying datasets (this may take a minute)..."

# Copy without Merchants datasets
cp data/FEwithoutMerchants/FE_test_without_merchants.csv data-docker/FEwithoutMerchants/
cp data/FEwithoutMerchants/FE_validation_without_merchants.csv data-docker/FEwithoutMerchants/
cp data/FEwithoutMerchants/FE_train_downsampled_1to5_without_merchants.csv data-docker/FEwithoutMerchants/
cp data/FEwithoutMerchants/FE_train_downsampled_1to10_without_merchants.csv data-docker/FEwithoutMerchants/

# Copy with Merchants datasets
cp data/FEwithMerchants/FE_test_with_merchants.csv data-docker/FEwithMerchants/
cp data/FEwithMerchants/FE_validation_with_merchants.csv data-docker/FEwithMerchants/
cp data/FEwithMerchants/FE_train_downsampled_1to5_with_merchants.csv data-docker/FEwithMerchants/
cp data/FEwithMerchants/FE_train_downsampled_1to10_with_merchants.csv data-docker/FEwithMerchants/

echo "Data preparation complete!"
echo "Data directory size:"
du -sh data-docker/

