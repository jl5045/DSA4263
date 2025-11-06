# Mobile Transaction Fraud Detection System

A comprehensive machine learning solution for detecting fraudulent financial mobile transactions, combining exploratory data analysis, feature engineering, and multiple machine learning models (Logistic Regression, Neural Networks, XGBoost, and Stacking Ensemble).

Data: [Kaggle Mobile Transaction Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Jupyter Notebooks Workflow](#jupyter-notebooks-workflow)
- [Streamlit Interactive Application](#streamlit-interactive-application)
- [Models & Performance](#models--performance)
- [Docker Deployment](#docker-deployment)
- [Production Deployment & System Integration](#production-deployment--system-integration)
- [Installation](#installation)
- [Usage](#usage)

---

## 🎯 Overview

This project implements an end-to-end fraud detection pipeline that:

- **Explores** financial transaction data with statistical analysis and visualization
- **Engineers** features to improve model performance
- **Trains** multiple machine learning models with different algorithms
- **Compares** model performance and selects the best approach
- **Deploys** an interactive Streamlit application for real-time predictions
- **Visualizes** fraud networks using graph analysis

### Key Features

- ✅ Multi-model comparison (Logistic Regression, Neural Networks, XGBoost, Ensemble)
- ✅ Imbalanced data handling with resampling techniques
- ✅ Interactive Streamlit dashboard for exploration and predictions
- ✅ Network graph visualization for fraud patterns
- ✅ Docker containerization for easy deployment
- ✅ Support for both merchant and non-merchant datasets

### System Architecture

The system follows a comprehensive pipeline from data ingestion through model deployment:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Data Sources & Processing                     │
├─────────────────────────────────────────────────────────────────┤
│  • Raw Financial Transactions                                   │
│  • Feature Engineering (Jupyter Notebooks)                      │
│  • Data Splitting & Resampling                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Model Training & Development                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ • Logistic Regression (Baseline)                         │   │
│  │ • Neural Networks (Deep Learning)                        │   │
│  │ • XGBoost (Gradient Boosting)                            │   │
│  │ • Stacking Ensemble (Meta-Learner)                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│          Interactive Streamlit Application                      │
├─────────────────────────────────────────────────────────────────┤
│  • EDA & Hypothesis Validation                                  │
│  • Feature Importance Analysis                                  │
│  • Model Comparisons & Results                                  │
│  • Live Fraud Prediction Interface                              │
│  • Network Visualization & Analytics                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Docker Containerization & Deployment               │
├─────────────────────────────────────────────────────────────────┤
│  • Containerized Application                                    │
│  • Volume Mounting for Data & Models                            │
│  • Resource-Optimized Configuration                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
DSA4263/
├── notebooks/                          # Jupyter notebooks for development
│   ├── 0_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 1a_train_test_val_split.ipynb  # Data splitting
│   ├── 1b_Resampling.ipynb            # Handling class imbalance
│   ├── 2_Feature_Engineering_Pipeline.py  # Feature engineering script
│   ├── 3_Logistic_Regression.ipynb    # Logistic Regression model
│   ├── 4_Neural_Network.ipynb         # Deep Learning model
│   ├── 5_XGBoost.ipynb                # Gradient Boosting model
│   ├── 6_Ensemble_Model.ipynb         # Stacking Ensemble model
│   └── ...                            # Additional experiments
│
├── src/                               # Streamlit application
│   ├── app.py                         # Main app entry point
│   └── pages/
│      ├── 2_EDA_Hypotheses.py       # Data exploration dashboard
│      ├── 3_Feature_Engineering_Hypotheses.py  # Feature analysis
│      ├── 4_Logistic_Regression_Results.py     # LR model results
│      ├── 5_Live_Fraud_Prediction.py           # Real-time predictions
│      ├── 6_Network_Explorer.py                # Fraud network viz
│      ├── 7_Model_Comparison.py                # Model performance
│      └── 7_Stacking_Ensemble_Model.py         # Ensemble results
│   
│
├── data/                              # Data directories
│   ├── raw/                           # Original datasets
│   ├── splits/                        # Train/test/val splits
│   └── FEwithMerchants/              # Feature-engineered data
│       └── FEwithoutMerchants/
│
├── models/                            # Trained model files
│   ├── XGB/                           # XGBoost models
│   ├── NN/                            # Neural Network models
│   └── validation_results_summary.csv
│
├── plots/                             # Generated visualizations for streamlit
│   ├── logistic_regression_metrics.csv
│   ├── neural_network_metrics.csv
│   ├── stacking_ensemble_metrics.csv
│   ├── fraud_network.html
|   └── Other plots generated from notebooks for streamlit display
│
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Docker compose setup
├── .dockerignore                      # Docker build exclusions
├── setup-docker-data.sh              # Data preparation script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip or conda
- (Optional) Docker and Docker Compose for containerized deployment

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jl5045/DSA4263.git
   cd DSA4263
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📓 Jupyter Notebooks Workflow

The notebook workflow is organized sequentially for a complete development pipeline:

### Phase 1: Data Preparation
- **`0_EDA.ipynb`** - Exploratory Data Analysis
  - Load and inspect raw data
  - Statistical summaries and distributions
  - Identify missing values and anomalies
  - Visualize fraud vs. legitimate transactions

- **`1a_train_test_val_split.ipynb`** - Data Splitting
  - Split data into train/validation/test sets
  - Maintain fraud class distribution across splits
  - Generate `data/splits/` CSV files

- **`1b_Resampling.ipynb`** - Handling Class Imbalance
  - Explore resampling techniques (oversampling, undersampling, SMOTE)
  - Create downsampled datasets (1:5 and 1:10 ratios)
  - Compare class distributions

### Phase 2: Feature Engineering
- **`2_Feature_Engineering_Pipeline.py`** - Feature Engineering Script
  - Create derived features from transaction attributes
  - Normalize and scale features
  - Select relevant features
  - Generate datasets in `data/FEwithMerchants/` and `data/FEwithoutMerchants/`

### Phase 3: Model Development

#### 3a. **`3_Logistic_Regression.ipynb`**
- Baseline model for comparison
- Train logistic regression classifier
- Evaluate with precision, recall, F1-score, ROC-AUC, PR-AUC
- Save metrics to `plots/logistic_regression_metrics.csv`

#### 3b. **`4_Neural_Network.ipynb`**
- Build and train deep neural network
- Experiment with different architectures
- Save trained model to `models/NN/NN_1to5_all.keras` and `NN_1to10_all.keras`
- Generate performance metrics

#### 3c. **`5_XGBoost.ipynb`**
- Gradient boosting implementation
- Hyperparameter tuning with cross-validation
- Save models to `models/XGB/best_xgb_1to5_*.json` and `best_xgb_1to10_*.json`
- Evaluate feature importance

#### 3d. **`6_Ensemble_Model.ipynb`**
- Stacking ensemble combining multiple models
- Meta-learner training
- Performance comparison with individual models
- Save results to `plots/stacking_ensemble_metrics.csv`

### Running Notebooks

**Option 1: Jupyter Lab**
```bash
jupyter lab
```
Navigate to `notebooks/` and open desired notebook.

**Option 2: VS Code**
- Install Jupyter extension in VS Code
- Open `.ipynb` files directly and run cells

---

## 🎨 Streamlit Interactive Application

The Streamlit app provides an interactive dashboard for exploring results and making predictions.

### Starting the App

```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`

### Available Pages

#### **EDA Hypotheses** (`2_EDA_Hypotheses.py`)
- Visual validation of key data hypotheses
- Transaction type fraud concentration analysis
- Transaction amount patterns for fraud vs. legitimate
- Top senders analysis
- Multicollinearity detection using VIF (Variance Inflation Factor)
- Displays precomputed plots from `plots/` directory

#### **Feature Engineering Hypotheses** (`3_Feature_Engineering_Hypotheses.py`)
- Visual validation of feature engineering decisions
- Feature importance analysis
- Feature correlation and relationship visualization
- Engineering approach effectiveness demonstration
- Displays precomputed plots from `plots/` directory

#### **Logistic Regression Results** (`4_Logistic_Regression_Results.py`)
- Baseline model performance metrics (F1-score, PR-AUC, ROC-AUC)
- Confusion matrix visualization
- ROC curve and PR curve comparisons
- Model interpretation for fraud detection
- Displays precomputed plots from `plots/` directory

#### **Neural Network Results** (`6_Neural_Network_Results.py`)
- Deep learning model performance metrics
- Training history and convergence analysis
- Confusion matrix and performance curves
- Complex pattern capture validation
- Displays precomputed plots from `plots/` directory

#### **Live Fraud Prediction** (`5_Live_Fraud_Prediction.py`)
- Interactive transaction input form (coming soon)
- Real-time fraud probability prediction
- Model confidence scores and explanations

#### **Network Explorer** (`6_Network_Explorer.py`)
- Interactive fraud network visualization
- Node connections between accounts and merchants
- Dataset selection (with/without merchants, various downsampling ratios)
- Graph analytics to identify fraud patterns
- Memory-efficient data loading with cache clearing option

#### **Model Comparison** (`7_Model_Comparison.py`)
- Side-by-side performance metrics for all models
- Overall model performance summary
- Model selection guidance
- Displays precomputed plots from `plots/` directory

#### **Stacking Ensemble Model** (`7_Stacking_Ensemble_Model.py`)
- Ensemble model combining multiple base models
- Individual model contributions analysis
- Performance comparison: Ensemble vs. individual models
- Meta-learner results and effectiveness
- Displays precomputed plots from `plots/` directory

---

## 🏆 Models & Performance

The project implements and compares four different ML approaches:

| Model | Type | Best Use Case | Key Metric |
|-------|------|---------------|-----------|
| **Logistic Regression** | Linear | Baseline, interpretability | Fast, simple, but poor accuracy |
| **Neural Network** | Deep Learning | Complex patterns | Mediocre accuracy |
| **XGBoost** | Gradient Boosting | Feature importance | High accuracy |
| **Stacking Ensemble** | Meta-Learner | Best overall | Combined strengths of Neural Network and XG Boost |

### Metrics Tracked
- **Precision:** False positive control
- **Recall:** False negative control
- **F1-Score:** Harmonic mean of precision/recall
- **ROC-AUC:** Overall discrimination ability
- **PR-AUC:** Precision-Recall tradeoff (better for imbalanced data)

**Main Metrics Tracked:**

- PR-AUC, with F1 score as the tie breaker
---

## 🐳 Docker Deployment

Deploy the Streamlit app in a containerized environment.

### Quick Start with Docker Compose
From root of project
1. **Prepare data locally** (data files won't be committed to GitHub):
   ```bash
   chmod +x setup-docker-data.sh
   ./setup-docker-data.sh
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the app:**
   - Open `http://localhost:8501`

### Manual Docker Build (instead of step 2 and 3)

```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/data-docker:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/plots:/app/plots \
  fraud-detection:latest
```

### Docker Configuration Details

- **Base Image:** Python 3.11 slim
- **Memory Limit:** 4GB (configurable in `docker-compose.yml`)
- **Volume Mounts:**
  - `data-docker/`: Feature-engineered datasets
  - `plots/`: Precomputed visualizations
- **Excluded from Docker Build:** Large raw datasets (see `.dockerignore`)

### Data Preparation Script

The `setup-docker-data.sh` script:
- Copies essential feature-engineered datasets to `data-docker/`
- Excludes large raw data files to keep image size manageable
- Runs locally to avoid GitHub file size limits

---

## 🌐 Production Deployment & System Integration

### Real-World System Architecture

This fraud detection system is designed to integrate into a comprehensive financial transaction monitoring pipeline:

![System Architecture Diagram](docs/architecture.png)

The diagram above illustrates how this fraud detection system integrates with production infrastructure:

#### **Data Ingestion & ETL Pipeline**
- **Data Sources:** Real-time transaction data from multiple channels
- **ETL Processing:** Snowflake for data warehousing and transformation
- **API Gateway:** AWS API Gateway routes incoming transaction requests
- **Lambda Functions:** AWS Lambda handles serverless compute for feature engineering and real-time processing

#### **Model Deployment Architecture**
- **Real-Time Fraud Detection:** VPC endpoints for low-latency fraud scoring
  - Incoming transactions → Lambda (data ingestion + feature engineering) → Model inference → Fraud decision
  - Blocks fraudulent transactions immediately at gateway level
- **Weekly Batch DAG Pipeline:** Scheduled weekly orchestration for comprehensive fraud analysis
  - Aggregates all transactions from the past week across multiple customers and merchants
  - Detects fraud patterns and fraud rings using network analysis
  - DynamoDB stores transaction parameters and customer data
  - Snowflake stores raw data and using ETL pipelines, send feature engineered data to S3 bucket
  - S3 buckets maintain engineered features
  - Sagemaker orchestrates weekly model retraining and performance monitoring
  - Generates fraud network visualizations for investigation teams in streamlit dashboard (or any other dashboard like PowerBI or Tableau)

#### **Dashboard & Visualization**
- **Analytics Dashboard:** Snowflake connects to visualization layer
- **Monitoring:** Real-time fraud detection metrics and model performance tracking
- **Actionable Insights:** Fraud network visualization and pattern analysis

---

## 📦 Dependencies

Key Python packages used in this project:

```
pandas              # Data manipulation
numpy               # Numerical computing
scikit-learn        # Machine learning algorithms
xgboost             # Gradient boosting
tensorflow          # Deep learning framework
streamlit           # Web application framework
plotly              # Interactive visualizations
pyvis               # Network graph visualization
networkx            # Network analysis
imbalanced-learn    # Resampling techniques
seaborn             # Statistical visualizations
matplotlib          # Plotting library
```

See `requirements.txt` for complete dependencies.

---

## 💡 Usage Examples

### Train a Model (Jupyter)
```python
# In notebook
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Load feature-engineered data
X_train = pd.read_csv('data/FEwithMerchants/FE_train_with_merchants.csv')

# Train model
model = GradientBoostingClassifier()
model.fit(X_train.drop('isFraud', axis=1), X_train['isFraud'])
```

### Make a Prediction (Streamlit App)
1. Navigate to "Live Fraud Prediction" page
2. Input transaction features
3. View fraud probability and confidence score

### Explore Network (Streamlit App)
1. Navigate to "Network Explorer" page
2. Interact with the fraud network visualization
3. Identify key merchants and patterns

---

## 🔍 Dataset Information

### Data Sources
- **Primary Dataset:** Synthetic financial transaction dataset
- **Fraud Rate:** Highly imbalanced (minority class fraud)
- **Features:** Transaction amount, merchant info, temporal features, etc.

### Data Variants
- **With Merchants:** Includes merchant category and name
- **Without Merchants:** Merchant features excluded
- **Downsampled:**
  - 1:5 ratio (1 fraud per 5 legitimate)
  - 1:10 ratio (1 fraud per 10 legitimate)

---

## 📊 Results & Metrics

Results are saved in the `plots/` and `models/` directories:

- `plots/logistic_regression_metrics.csv` - LR performance
- `plots/neural_network_metrics.csv` - NN performance
- `plots/stacking_ensemble_metrics.csv` - Ensemble performance
- `plots/fraud_network.html` - Interactive network visualization
- `models/validation_results_summary.csv` - Cross-validation summary

---

**Last Updated:** November 2025


