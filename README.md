# Machine Learning - ElevvoPathways

A collection of machine learning projects and tasks for skill development and learning.

## 📊 Task 1: Student Performance Indicator

A complete machine learning project that predicts student exam scores using various performance factors.

### 🎯 Project Overview
- **Dataset**: Student performance factors including study hours, attendance, parental involvement, etc.
- **Goal**: Build a regression model to predict exam scores
- **Approach**: Linear regression with comprehensive feature analysis and visualization
- **Tools**: Python, pandas, scikit-learn, matplotlib, seaborn

### 💻 Code Implementation

#### Data Loading and Preprocessing
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv('data/StudentPerformanceFactors.csv')

# Data cleaning and preprocessing
for col in df.columns:
    if df[col].dtype == 'object':
        converted = pd.to_numeric(df[col], errors='ignore')
        if not converted.equals(df[col]):
            df[col] = converted

# Handle missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        if df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
```

#### Exploratory Data Analysis
```python
# Target variable analysis
target = 'Exam_Score'
print(f"Score range: {df[target].min():.2f} - {df[target].max():.2f}")
print(f"Score average: {df[target].mean():.2f}")

# Correlation analysis
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(df[target], bins=20, alpha=0.7)
plt.title(f'Distribution of {target}')

plt.subplot(1, 3, 2)
plt.scatter(df['Hours_Studied'], df[target], alpha=0.6)
plt.title('Study Hours vs Exam Score')

plt.subplot(1, 3, 3)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

#### Model Training and Evaluation
```python
# Prepare features and target
X = df.select_dtypes(include=[np.number]).drop(columns=[target])
y = df[target]

# Handle categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.select_dtypes(include=[np.number]).drop(columns=[target])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
```

### 📈 Results & Visualizations

#### Data Analysis Overview
![Analysis Overview](Task-1-student-performance-indicator/assets/plot1.png)

*Distribution analysis, correlation matrix, and key relationships*

#### Model Performance
![Model Performance](Task-1-student-performance-indicator/assets/plot2.png)

*Model predictions vs actual values and performance metrics*

### 🔍 Key Features Analyzed
- Study hours per week
- Attendance rate
- Parental involvement
- Access to resources
- Extracurricular activities
- Sleep hours
- Previous scores
- Motivation level
- Internet access
- Tutoring sessions
- Family income
- Teacher quality
- School type
- Peer influence
- Physical activity
- Learning disabilities
- Parental education level
- Distance from home
- Gender

### 📊 Model Performance Metrics
- **R² Score**: Measures how well the model explains the variance
- **RMSE**: Root Mean Square Error for prediction accuracy
- **MAE**: Mean Absolute Error for average prediction difference

### 🛠️ Technologies Used
- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning library

### 📝 Key Insights
The analysis reveals important factors that influence student performance:
1. **Study Hours**: Strong positive correlation with exam scores
2. **Attendance**: Regular attendance significantly impacts performance
3. **Parental Involvement**: Supportive parents boost student outcomes
4. **Resource Access**: Better resources lead to better performance

### 🎓 Learning Outcomes
- Data preprocessing and cleaning techniques
- Exploratory data analysis (EDA)
- Feature importance analysis
- Linear regression modeling
- Model evaluation and validation
- Data visualization best practices

---

## 👨‍💻 Author

**Abdelkader**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abdelkader7151)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdelrhman-abdelkader-6313a4291/)

---

*This project is part of the ElevvoPathways Machine Learning curriculum.*
