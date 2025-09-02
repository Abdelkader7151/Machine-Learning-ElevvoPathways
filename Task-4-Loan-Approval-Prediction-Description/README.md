## Task 4 – Loan Approval Prediction

This repo contains a complete, runnable pipeline to train a classifier that predicts loan approval. The pipeline handles missing values, encodes categorical features, uses SMOTE to address class imbalance, and evaluates with precision/recall/F1. It also saves plots and artifacts in `outputs`.

### Quick start (Windows PowerShell)

Run the setup script. It creates a virtual environment, installs dependencies, and runs the training script. If no dataset is present, a small public sample is downloaded automatically.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1
```

Optionally provide your own CSV dataset path and target column:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1 -DataPath "data\\your_loan_data.csv"
# Or run manually:
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --data data\your_loan_data.csv --target Loan_Status --outputs outputs
```

### Use the official Kaggle dataset (recommended)

If you have a Kaggle account, you can download the "Loan-Approval-Prediction-Dataset" directly using the Kaggle API.

1) Configure Kaggle credentials once (only needed the first time):

```powershell
# Download your API token from https://www.kaggle.com/settings/account
# Save kaggle.json to %USERPROFILE%\.kaggle\kaggle.json
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null
Copy-Item -Path "C:\Path\To\kaggle.json" -Destination "$env:USERPROFILE\.kaggle\kaggle.json" -Force
```

2) Run setup and training with Kaggle download enabled:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1 -UseKaggle
# Or manually
python main.py --use-kaggle --kaggle-dataset ajay1735/loan-approval-prediction --outputs outputs
```

This will download the Kaggle dataset and save a CSV at `data/loan_approval.csv` before training.

### Expected outputs (inside `outputs/`)
- `class_distribution.png`
- `<model>_confusion_matrix.png`
- `<model>_report.txt` (precision, recall, F1)
- `<model>_model.joblib` (trained pipeline)
- `<model>_feature_importance.csv` and `.png` (if supported)

> Note: `<model>` will be `logreg` or `rf`, whichever scores best via cross-validated F1 on the training set.

### Dataset
- Default: If `data/loan_approval.csv` is missing, the script downloads a small public sample from `ybifoundation` GitHub for convenience.
- Recommended: You can supply the Kaggle “Loan-Approval-Prediction-Dataset” CSV. If the target column differs, pass `--target`.

### What the pipeline does
1. Loads CSV and infers the target column (defaults to `Loan_Status` if present).
2. Drops obvious ID columns.
3. Splits train/test with stratification.
4. Preprocesses:
   - Numeric: median impute, `StandardScaler`
   - Categorical: most-frequent impute, `OneHotEncoder`
5. Uses `SMOTE` to balance classes during training.
6. Trains and cross-validates Logistic Regression and Random Forest, selects the best by F1.
7. Evaluates on the test set, saves confusion matrix and classification report.
8. Saves trained pipeline and feature importances (if available).

### Reproducibility
The pipeline uses fixed random seeds for splitting, SMOTE, and models.

### Troubleshooting
- If you are behind a proxy or have restricted internet, place your dataset at `data/loan_approval.csv` and re-run.
- If feature importance files are not generated, the selected model may not support them. Random Forest and Logistic Regression are supported.
# Forest Cover Type Classification

A comprehensive machine learning project that predicts forest cover types based on cartographic and environmental features using the Covertype dataset from UCI. This project implements multi-class classification with Random Forest and XGBoost models, including advanced model comparison and hyperparameter tuning.

## 🎯 Project Objectives

- **Predict forest cover types** using cartographic and environmental features
- **Clean and preprocess data** including categorical variable handling
- **Train multi-class classification models** (Random Forest, XGBoost)
- **Evaluate model performance** with confusion matrix and metrics
- **Analyze feature importance** to understand key predictors
- **Compare different models** (Random Forest vs XGBoost)
- **Perform hyperparameter tuning** for optimal performance

## 📁 Project Structure
- `main.py`: Complete end-to-end classification pipeline
- `requirements.txt`: All required Python dependencies
- `data/`: Dataset storage (auto-downloads Covertype dataset)
- `outputs/`: Generated visualizations, models, and analysis results

## 📊 Dataset
- **Source**: Covertype Dataset (UCI Machine Learning Repository)
- **File**: `data/covtype.data`
- **Features**: 54 cartographic and environmental variables
- **Target**: 7 forest cover types
- **Samples**: 581,012 instances
- **Auto-download**: Script automatically downloads dataset if missing

## 🌲 Forest Cover Types
1. **Spruce/Fir**
2. **Lodgepole Pine**
3. **Ponderosa Pine**
4. **Cottonwood/Willow**
5. **Aspen**
6. **Douglas-fir**
7. **Krummholz**

## 🚀 Quick Start

### Setup Environment
```powershell
# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run Analysis
```powershell
# Basic analysis with Random Forest
python main.py

# Advanced analysis with model comparison and tuning
python main.py --compare-models --tune-hyperparameters
```

## 📈 Analysis Pipeline

1. **Data Loading**: Automatic Covertype dataset download and loading
2. **Data Preprocessing**: Feature scaling, categorical encoding, data cleaning
3. **Exploratory Analysis**: Feature distributions and correlations
4. **Model Training**: Random Forest and XGBoost classification
5. **Model Evaluation**: Accuracy, precision, recall, F1-score metrics
6. **Confusion Matrix**: Multi-class performance visualization
7. **Feature Importance**: Identify most predictive variables
8. **Model Comparison**: Random Forest vs XGBoost performance
9. **Hyperparameter Tuning**: Grid search for optimal parameters

## 📊 Generated Outputs

### Visualizations
- `outputs/confusion_matrix.png`: Multi-class confusion matrix
- `outputs/feature_importance.png`: Top features ranked by importance
- `outputs/model_comparison.png`: Performance comparison between models
- `outputs/class_distribution.png`: Target class distribution analysis

### Analysis Files
- `outputs/classification_report.txt`: Detailed performance metrics
- `outputs/best_model.joblib`: Trained model with best performance
- `outputs/feature_rankings.csv`: Complete feature importance rankings
- Console output: Model performance, accuracy scores, and insights

## 🛠️ Command Line Options

- `--compare-models`: Enable Random Forest vs XGBoost comparison
- `--tune-hyperparameters`: Perform grid search hyperparameter tuning
- `--test-size`: Train/test split ratio (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

## 🔧 Troubleshooting

- **Dataset issues**: If auto-download fails, manually place `covtype.data` in `data/` folder
- **Memory issues**: Reduce dataset size with sampling for initial testing
- **Performance issues**: Use smaller parameter grids for hyperparameter tuning
- **Visualization issues**: Check `outputs/` folder for saved plots

## 🎯 Expected Results

The analysis will classify forest cover types with expected performance:
- **Random Forest**: ~85-90% accuracy on test set
- **XGBoost**: ~87-92% accuracy on test set
- **Key Features**: Elevation, aspect, slope, and wilderness areas
- **Best Performance**: XGBoost typically outperforms Random Forest

## 📚 Machine Learning Concepts Covered

- **Multi-class Classification**: Predicting among 7 forest cover types
- **Tree-based Modeling**: Random Forest and XGBoost algorithms
- **Feature Engineering**: Handling categorical and numerical features
- **Model Evaluation**: Confusion matrix, classification metrics
- **Hyperparameter Tuning**: Grid search optimization
- **Model Comparison**: Systematic performance evaluation