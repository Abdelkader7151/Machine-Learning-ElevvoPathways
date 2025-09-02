#!/usr/bin/env python3
"""
Forest Cover Type Classification - Complete Solution
Predicts forest cover types using cartographic and environmental features
"""

import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Forest cover type names
COVER_TYPES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine", 
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

def ensure_dirs() -> None:
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    """Load the Covertype dataset directly from UCI via ucimlrepo."""
    print("📥 Fetching Covertype dataset from UCI ML Repository (id=31)...")
    covertype = fetch_ucirepo(id=31)

    # Data as pandas DataFrames
    X: pd.DataFrame = covertype.data.features
    y: pd.DataFrame = covertype.data.targets

    # Optional: show metadata and variable info (as requested)
    print("\n📋 Dataset Metadata:")
    print(covertype.metadata)
    print("\n📊 Variable Information:")
    print(covertype.variables)

    # Ensure target column is named 'Cover_Type' for downstream code
    target_col = y.columns[0]
    if target_col != 'Cover_Type':
        y = y.rename(columns={target_col: 'Cover_Type'})

    df = pd.concat([X, y], axis=1)
    print(f"✅ Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis."""
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Target distribution (use original labels for display)
    print("\n=== Cover Type Distribution ===")
    cover_counts = df['Cover_Type'].value_counts().sort_index()
    for cover_type, count in cover_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{cover_type}. {COVER_TYPES[cover_type]}: {count:,} ({percentage:.1f}%)")

    # Create class distribution plot
    plt.figure(figsize=(10, 6))
    cover_counts.plot(kind='bar')
    plt.title('Forest Cover Type Distribution')
    plt.xlabel('Cover Type')
    plt.ylabel('Count')
    plt.xticks(range(len(COVER_TYPES)),
               [f"{i+1}. {COVER_TYPES[i+1]}" for i in range(len(COVER_TYPES))],
               rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the data for modeling."""
    print("\n=== Data Preprocessing ===")

    # Separate features and target
    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']

    # XGBoost requires classes starting from 0, so adjust labels
    # Store original labels for display purposes
    y_adjusted = y - 1  # Convert [1,2,3,4,5,6,7] to [0,1,2,3,4,5,6]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_adjusted, test_size=test_size, random_state=random_state, stratify=y_adjusted
    )

    # Scale the continuous features (first 10 features)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    continuous_features = X_train.columns[:10]  # First 10 are continuous
    X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

    print(f"✅ Data split: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test samples")
    print(f"✅ Features scaled: {len(continuous_features)} continuous features")
    print("✅ Class labels adjusted for XGBoost compatibility")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                       tune_hyperparameters: bool = False) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    print("\n=== Training Random Forest ===")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            random_state=42, 
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        return rf


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 tune_hyperparameters: bool = False) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    print("\n=== Training XGBoost ===")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                  model_name: str) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n=== {model_name} Performance ===")
    print(f"Accuracy: {accuracy:.4f}")

    # Convert predictions back to original labels for display
    y_pred_display = y_pred + 1  # Convert back from [0,1,2,3,4,5,6] to [1,2,3,4,5,6,7]
    y_test_display = y_test + 1

    # Detailed classification report
    report = classification_report(y_test_display, y_pred_display, target_names=list(COVER_TYPES.values()))
    print(report)

    # Save classification report
    with open(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_report.txt", 'w') as f:
        f.write(f"{model_name} Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    return {
        'accuracy': accuracy,
        'predictions': y_pred_display  # Return converted predictions for confusion matrix
    }


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(COVER_TYPES.values()),
                yticklabels=list(COVER_TYPES.values()))
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Cover Type')
    plt.ylabel('Actual Cover Type')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model: Any, feature_names: list, model_name: str) -> None:
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        # Create feature importance dataframe
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        feature_imp.to_csv(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv", 
                          index=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 12))
        top_features = feature_imp.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_feature_importance.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Feature importance saved for {model_name}")
        print("Top 10 most important features:")
        for i, (_, row) in enumerate(top_features.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")


def compare_models(results: Dict[str, Dict]) -> None:
    """Compare model performance."""
    print("\n=== Model Comparison ===")
    
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print comparison
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"🏆 Best performing model: {best_model}")
    for name in model_names:
        print(f"{name}: {results[name]['accuracy']:.4f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forest Cover Type Classification")
    parser.add_argument("--compare-models", action="store_true",
                       help="Compare Random Forest vs XGBoost")
    parser.add_argument("--tune-hyperparameters", action="store_true",
                       help="Perform hyperparameter tuning")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility")
    return parser.parse_args()


def main():
    """Main execution function."""
    print("🌲 Forest Cover Type Classification Analysis 🌲")
    print("=" * 55)
    
    args = parse_args()
    ensure_dirs()
    
    # Load and explore data
    df = load_data()
    explore_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(
        df, test_size=args.test_size, random_state=args.random_state
    )

    feature_names = df.drop('Cover_Type', axis=1).columns.tolist()
    results = {}

        # Convert test labels back to original format for display
    y_test_original = y_test + 1

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, args.tune_hyperparameters)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    plot_confusion_matrix(y_test_original, rf_results['predictions'], "Random Forest")
    plot_feature_importance(rf_model, feature_names, "Random Forest")
    results["Random Forest"] = rf_results

    # Train XGBoost if comparison requested
    if args.compare_models:
        xgb_model = train_xgboost(X_train, y_train, args.tune_hyperparameters)
        xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        plot_confusion_matrix(y_test_original, xgb_results['predictions'], "XGBoost")
        plot_feature_importance(xgb_model, feature_names, "XGBoost")
        results["XGBoost"] = xgb_results
        
        # Compare models
        compare_models(results)
        
        # Save best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = xgb_model if best_model_name == "XGBoost" else rf_model
        joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")
        print(f"✅ Best model ({best_model_name}) saved to outputs/best_model.joblib")
    else:
        # Save Random Forest model
        joblib.dump(rf_model, OUTPUT_DIR / "random_forest_model.joblib")
        print("✅ Random Forest model saved to outputs/random_forest_model.joblib")
    
    print(f"\n🎉 Analysis Complete! Check the 'outputs/' folder for results.")
    print(f"📊 Generated files:")
    for file in OUTPUT_DIR.glob("*"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()