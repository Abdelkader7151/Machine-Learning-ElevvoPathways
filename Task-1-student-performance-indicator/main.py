# Student Score Prediction - Complete Solution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def _find_dataset_path() -> Optional[Tuple[str, str]]:
	"""Return (path, format) where format is 'csv' or 'excel', or None if not found."""
	data_dir = 'data'
	preferred = os.path.join(data_dir, 'Student_Performance_Factors.csv')
	if os.path.exists(preferred):
		return preferred, 'csv'

	if not os.path.isdir(data_dir):
		return None

	# Collect candidate files from data/ recursively
	candidates = []
	for root, _, files in os.walk(data_dir):
		for name in files:
			lower = name.lower()
			if lower.endswith('.csv'):
				candidates.append((os.path.join(root, name), 'csv', lower))
			elif lower.endswith('.xlsx') or lower.endswith('.xls'):
				candidates.append((os.path.join(root, name), 'excel', lower))

	if not candidates:
		return None

	# Prefer files whose names include these keywords
	keywords = ['student', 'performance', 'score', 'exam']
	def score(name_lower: str) -> int:
		return sum(1 for k in keywords if k in name_lower)

	best = max(candidates, key=lambda t: (score(t[2]), t[2]))
	return best[0], best[1]

def _infer_target_column(df: pd.DataFrame) -> Optional[str]:
	"""Infer a likely target column. Prefer columns named like score/mark/grade, else Exam_Score, else first numeric."""
	name_priority = ['exam_score', 'score', 'marks', 'mark', 'grade', 'final', 'result']
	lower_map = {c.lower(): c for c in df.columns}
	for key in name_priority:
		for col_lower, original in lower_map.items():
			if key in col_lower:
				if pd.api.types.is_numeric_dtype(df[original]):
					return original
	# Fallback to explicit Exam_Score if exists but non-numeric
	if 'Exam_Score' in df.columns:
		return 'Exam_Score'
	# Fallback: any numeric column
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	return numeric_cols[0] if numeric_cols else None

def main():
	print("=== Student Score Prediction Analysis ===")
	
	# Step 1: Load Dataset with robust detection
	found = _find_dataset_path()
	if not found:
		print("❌ Dataset not found. Put your CSV/XLSX in the 'data/' folder (e.g., Student_Performance_Factors.csv)")
		return
	path, fmt = found
	try:
		if fmt == 'csv':
			df = pd.read_csv(path)
		else:
			df = pd.read_excel(path)
		print(f"✅ Dataset loaded: {path}")
		print(f"Dataset shape: {df.shape}")
	except Exception as exc:
		print(f"❌ Failed to load dataset at {path}: {exc}")
		return
	
	# Step 2: Explore Dataset
	print("\n=== Dataset Overview ===")
	print("Columns:", df.columns.tolist())
	print("\nFirst few rows:")
	print(df.head())
	print("\nDataset Info:")
	df.info()
	
	# Step 3: Data Cleaning
	print("\n=== Data Cleaning ===")
	print("Missing values before cleaning:")
	print(df.isnull().sum())
	
	# Try to convert numeric-like text columns to numbers
	for col in df.columns:
		if df[col].dtype == 'object':
			converted = pd.to_numeric(df[col], errors='ignore')
			if not converted.equals(df[col]):
				df[col] = converted
	
	# Impute missing values: numeric with median, categorical with mode
	for col in df.columns:
		if pd.api.types.is_numeric_dtype(df[col]):
			df[col] = df[col].fillna(df[col].median())
		else:
			if df[col].isnull().any():
				mode_val = df[col].mode(dropna=True)
				if not mode_val.empty:
					df[col] = df[col].fillna(mode_val.iloc[0])

	print("Missing values after cleaning:")
	print(df.isnull().sum())
	
	# Step 4: Basic Analysis
	if 'Exam_Score' in df.columns:
		target = 'Exam_Score'
	else:
		target = _infer_target_column(df)
		if not target:
			print("❌ Could not infer a target column (no numeric columns found).")
			return
		print(f"Auto-selected target column: {target}")
	
	# Step 5: Feature Analysis
	print(f"\n=== Analyzing {target} ===")
	print(f"Score range: {df[target].min():.2f} - {df[target].max():.2f}")
	print(f"Score average: {df[target].mean():.2f}")
	
	# Step 6: Simple Visualization
	plt.figure(figsize=(12, 4))
	
	plt.subplot(1, 3, 1)
	plt.hist(df[target], bins=20, alpha=0.7)
	plt.title(f'Distribution of {target}')
	plt.xlabel(target)
	plt.ylabel('Frequency')
	
	# Find study hours column
	study_col = None
	for col in df.columns:
		if 'hour' in col.lower() and 'stud' in col.lower():
			study_col = col
			break
	
	if study_col:
		plt.subplot(1, 3, 2)
		plt.scatter(df[study_col], df[target], alpha=0.6)
		plt.title(f'{study_col} vs {target}')
		plt.xlabel(study_col)
		plt.ylabel(target)
	
	plt.subplot(1, 3, 3)
	# Correlation heatmap (organized by correlation strength with target)
	numeric_df = df.select_dtypes(include=[np.number])
	if numeric_df.shape[1] >= 2:
		corr = numeric_df.corr(numeric_only=True)
		# Order by absolute correlation with target (keep target last)
		if target in corr.columns:
			order = corr[target].abs().sort_values(ascending=False).index.tolist()
			# Keep up to 8 most informative to avoid clutter
			top_n = min(8, len(order))
			order = order[:top_n]
			if target in order:
				order = [c for c in order if c != target] + [target]
			corr = corr.loc[order, order]
		sns.heatmap(
			corr,
			annot=True,
			fmt=".2f",
			cmap='coolwarm',
			vmin=-1,
			vmax=1,
			center=0,
			square=True,
			linewidths=0.5,
			linecolor='lightgray',
			cbar_kws={"shrink": 0.7}
		)
		plt.xticks(rotation=45, ha='right')
		plt.yticks(rotation=0)
		plt.title('Correlation Matrix')
	else:
		plt.text(0.5, 0.5, 'Not enough numeric columns for correlation', ha='center', va='center')
		plt.axis('off')
	
	plt.tight_layout()
	
	# Save the plot
	os.makedirs('assets', exist_ok=True)
	plt.savefig('assets/analysis_overview.png', dpi=300, bbox_inches='tight')
	print("📊 Analysis plots saved to assets/analysis_overview.png")
	plt.show()
	
	# Step 7: Simple Model
	print("\n=== Building Simple Model ===")
	
	# Prepare features
	numeric_features = df.select_dtypes(include=[np.number])
	if target in numeric_features.columns:
		X = numeric_features.drop(columns=[target])
	else:
		# target might be non-numeric (e.g., stored as text); try to coerce
		df[target] = pd.to_numeric(df[target], errors='coerce')
		df[target] = df[target].fillna(df[target].median())
		numeric_features = df.select_dtypes(include=[np.number])
		X = numeric_features.drop(columns=[target], errors='ignore')
	y = df[target]
	
	# Handle categorical variables if any
	categorical_cols = df.select_dtypes(include=['object']).columns
	if len(categorical_cols) > 0:
		print(f"Converting categorical columns: {categorical_cols.tolist()}")
		df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
		X = df_encoded.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
		y = df_encoded[target]

	# Ensure there is at least one feature
	if X.shape[1] == 0:
		print("❌ No usable numeric features found after preprocessing.")
		return
	
	# Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Train model
	model = LinearRegression()
	model.fit(X_train, y_train)
	
	# Make predictions
	y_pred = model.predict(X_test)
	
	# Evaluate
	r2 = r2_score(y_test, y_pred)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	mae = mean_absolute_error(y_test, y_pred)
	
	print(f"✅ Model Performance:")
	print(f"R² Score: {r2:.4f}")
	print(f"RMSE: {rmse:.4f}")
	print(f"MAE: {mae:.4f}")
	
	# Feature importance
	feature_importance = pd.DataFrame({
		'feature': X.columns,
		'importance': abs(model.coef_)
	}).sort_values('importance', ascending=False)
	
	print(f"\nTop 5 Most Important Features:")
	print(feature_importance.head())

	# Visualize predictions vs actual and residuals
	plt.figure(figsize=(10, 4))
	plt.subplot(1, 2, 1)
	plt.scatter(y_test, y_pred, alpha=0.6)
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	plt.title('Actual vs Predicted')
	min_val = min(y_test.min(), y_pred.min())
	max_val = max(y_test.max(), y_pred.max())
	plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

	plt.subplot(1, 2, 2)
	residuals = y_test - y_pred
	plt.hist(residuals, bins=20, alpha=0.7)
	plt.title('Residuals Distribution')
	plt.xlabel('Residual')
	plt.ylabel('Frequency')
	plt.tight_layout()
	
	# Save the predictions plot
	plt.savefig('assets/model_predictions.png', dpi=300, bbox_inches='tight')
	print("📊 Model predictions plot saved to assets/model_predictions.png")
	plt.show()

	# Optional: Polynomial regression on study hours if available
	if study_col and pd.api.types.is_numeric_dtype(df[study_col]):
		print("\n=== Polynomial Regression (bonus) on study hours ===")
		poly = PolynomialFeatures(degree=2, include_bias=False)
		study_vals = df[[study_col]].values
		target_vals = df[target].values
		X_poly = poly.fit_transform(study_vals)
		Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_poly, target_vals, test_size=0.2, random_state=42)
		poly_model = LinearRegression()
		poly_model.fit(Xp_train, yp_train)
		yp_pred = poly_model.predict(Xp_test)
		print(f"R² (poly degree=2): {r2_score(yp_test, yp_pred):.4f}")
		print(f"RMSE (poly degree=2): {np.sqrt(mean_squared_error(yp_test, yp_pred)):.4f}")
	
	print("\n=== Analysis Complete! ===")

if __name__ == "__main__":
	main()


