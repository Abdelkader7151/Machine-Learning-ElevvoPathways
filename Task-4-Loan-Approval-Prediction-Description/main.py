import os
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline as SkPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def ensure_directories(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please ensure the CSV file exists."
        )

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def infer_target_column(df: pd.DataFrame, preferred_target: str | None) -> str:
    if preferred_target and preferred_target in df.columns:
        return preferred_target
    # Common target names for loan approval problems
    for candidate in ["Loan_Status", "loan_status", "Status", "Approved"]:
        if candidate in df.columns:
            return candidate
    # Fallback: last column
    return df.columns[-1]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop ID-like columns if present
    id_like = [
        c
        for c in df.columns
        if c.lower() in {"loan_id", "id"} or c.lower().endswith("_id") or c.lower().endswith("id")
    ]
    if id_like:
        df = df.drop(columns=id_like)
        print(f"Dropped ID columns: {id_like}")
    return df


def prepare_features(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y = df[target_col]

    # Normalize typical labels like 'Y'/'N' to 1/0
    if y.dtype == object:
        y = y.str.strip().str.upper().map({"Y": 1, "N": 0}).fillna(y)
    # Coerce to numeric if still not numeric
    if not np.issubdtype(y.dtype, np.number):
        y = pd.Categorical(y).codes

    X = df.drop(columns=[target_col])

    categorical_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("category")]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    print(f"Features: {len(X.columns)} total ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    
    return X, y.astype(int), numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Support scikit-learn >=1.2 (sparse_output) and older (sparse)
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    model_name: str,
) -> None:
    report = classification_report(y_true, y_pred, digits=4)
    report_path = os.path.join(out_dir, f"{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")


def plot_class_distribution(y: pd.Series, out_dir: str) -> None:
    values = pd.Series(y).value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(values.index.astype(str), values.values, color=['lightcoral', 'skyblue'])
    plt.title("Class Distribution")
    plt.xlabel("Class (0=Rejected, 1=Approved)")
    plt.ylabel("Count")
    
    # Add count labels on bars
    for bar, count in zip(bars, values.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    path = os.path.join(out_dir, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution to {path}")


def extract_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    try:
        names = list(fitted_preprocessor.get_feature_names_out())
        # Clean up names like 'num__Feature' / 'cat__onehot__Category'
        names = [n.split("__", 1)[-1] for n in names]
        return names
    except Exception:
        return []


def save_feature_importance(
    pipeline: Pipeline, out_dir: str, model_name: str
) -> None:
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    feature_names = extract_feature_names(preprocessor)

    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        values = np.abs(coef)  # Use absolute values for logistic regression

    if values is None or len(feature_names) == 0:
        print("Feature importance not available for this model.")
        return

    importances = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(values).ravel()}
    ).sort_values("importance", ascending=False)

    csv_path = os.path.join(out_dir, f"{model_name}_feature_importance.csv")
    importances.to_csv(csv_path, index=False)
    print(f"Saved feature importance CSV to {csv_path}")

    top_n = min(15, len(importances))
    plt.figure(figsize=(10, 8))
    sns.barplot(
        y=importances.head(top_n)["feature"],
        x=importances.head(top_n)["importance"],
        orient="h",
        palette="viridis"
    )
    plt.title(f"Top {top_n} Feature Importance - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("")
    fig_path = os.path.join(out_dir, f"{model_name}_feature_importance.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to {fig_path}")


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: str,
    random_state: int = 42,
) -> None:
    ensure_directories(out_dir)
    plot_class_distribution(y, out_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    numeric_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    candidates = {
        "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state),
    }

    best_model_name = None
    best_cv_score = -np.inf

    print("\nTraining and evaluating models...")
    for name, model in candidates.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                ("model", model),
            ]
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        # Suppress warnings from some scorers on constant predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        mean_score = float(np.mean(scores))
        print(f"Model {name}: CV F1 = {mean_score:.4f}")
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_model_name = name

    assert best_model_name is not None
    print(f"\nSelected model: {best_model_name} (CV F1={best_cv_score:.4f})")

    best_model = candidates[best_model_name]
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", best_model),
        ]
    )
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    evaluate_and_save(y_test, y_pred, out_dir, best_model_name)

    # Save model
    model_path = os.path.join(out_dir, f"{best_model_name}_model.joblib")
    dump(best_pipeline, model_path)
    print(f"Saved trained pipeline to {model_path}")

    # Save feature importance if available
    save_feature_importance(best_pipeline, out_dir, best_model_name)
    
    print(f"\n✅ Training completed! Check '{out_dir}' folder for all outputs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Loan Approval Prediction Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/loan_approval_dataset.csv",
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name. If omitted, it is inferred (defaults to Loan_Status if present).",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="outputs",
        help="Directory to write outputs",
    )

    args = parser.parse_args()

    ensure_directories(args.outputs)

    print("🏦 Loan Approval Prediction Pipeline")
    print("=" * 40)
    
    df = load_dataset(args.data)
    df = clean_dataframe(df)
    target_col = infer_target_column(df, args.target)
    print(f"Using target column: {target_col}")

    X, y, _, _ = prepare_features(df, target_col)
    
    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Class distribution: {class_counts}")
    
    train_and_evaluate(X, y, args.outputs)


if __name__ == "__main__":
    main()