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