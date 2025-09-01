# Machine Learning - ElevvoPathways

A collection of machine learning projects and tasks for skill development and learning.

## 📊 Task 1: Student Performance Indicator

A complete machine learning project that predicts student exam scores using various performance factors.

### 🎯 Overview
- **Dataset**: Student performance factors including study hours, attendance, parental involvement, etc.
- **Goal**: Build a regression model to predict exam scores
- **Approach**: Linear regression with comprehensive feature analysis and visualization
- **Tools**: Python, pandas, scikit-learn, matplotlib, seaborn

### 📋 Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

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

![Analysis Overview](Task-1-student-performance-indicator/assets/plot1.png)
*Distribution analysis, correlation matrix, and key relationships*

![Model Performance](Task-1-student-performance-indicator/assets/plot2.png)
*Model predictions vs actual values and performance metrics*

### 📊 Performance Metrics
- **R² Score**: Measures how well the model explains the variance
- **RMSE**: Root Mean Square Error for prediction accuracy  
- **MAE**: Mean Absolute Error for average prediction difference

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

### 📝 Key Insights
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

## 🎯 Task 2: Customer Segmentation

A comprehensive clustering analysis project that segments mall customers based on income and spending patterns using K-Means and DBSCAN algorithms.

### 🎯 Overview
- **Dataset**: Mall Customers with Annual Income and Spending Score data
- **Goal**: Segment customers into distinct groups for targeted marketing
- **Approach**: K-Means clustering with optimal cluster selection using Elbow Method and Silhouette Analysis
- **Tools**: Python, pandas, scikit-learn, matplotlib, seaborn
- **Bonus**: DBSCAN clustering comparison and business insights

### 📋 Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 💻 Code Implementation

#### Customer Data Loading and Preprocessing
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Mall Customers dataset
df = pd.read_csv('data/Mall_Customers.csv')

# Select features for clustering
income_col = 'Annual Income (k$)'
spending_col = 'Spending Score (1-100)'
X = df[[income_col, spending_col]].copy()

# Feature scaling for optimal clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### Optimal Cluster Selection
```python
# Elbow Method and Silhouette Analysis
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Find optimal k (highest silhouette score)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.title('Elbow Method For Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.title('Silhouette Score For Different k')
plt.show()
```

#### K-Means Clustering and Visualization
```python
# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(X[income_col], X[spending_col], 
                     c=cluster_labels, cmap='viridis', alpha=0.7)
# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

# DBSCAN comparison
plt.subplot(1, 2, 2)
dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
plt.scatter(X[income_col], X[spending_col], 
           c=dbscan_labels, cmap='plasma', alpha=0.7)
plt.title('DBSCAN Clustering')
plt.show()

# Business insights
for i in range(optimal_k):
    cluster_data = X[X['Cluster'] == i]
    print(f"Cluster {i}: {len(cluster_data)} customers")
    print(f"  Avg Income: ${cluster_data[income_col].mean():.1f}k")
    print(f"  Avg Spending: {cluster_data[spending_col].mean():.1f}")
```

### 📈 Results & Visualizations

![Data Exploration](Task-2-Customer-Segmentation/assets/data_exploration.png)
*Customer income and spending score distributions with scatter plot*

![Cluster Optimization](Task-2-Customer-Segmentation/assets/cluster_optimization.png)
*Elbow Method and Silhouette Analysis for optimal cluster selection*

![Clustering Results](Task-2-Customer-Segmentation/assets/clustering_results.png)
*K-Means clustering results with centroids and DBSCAN comparison*

### 📊 Performance Metrics
- **Silhouette Score**: Measures cluster separation and cohesion (-1 to 1, higher is better)
- **Inertia**: Within-cluster sum of squared distances (lower is better for compact clusters)
- **Elbow Method**: Visual technique to find optimal number of clusters
- **Business Impact**: Customer segment identification for targeted marketing strategies

### 🔍 Key Features Analyzed
- **Annual Income (k$)**: Customer's yearly income in thousands
- **Spending Score (1-100)**: Mall-assigned score based on customer behavior and spending patterns
- **Derived Segments**: Premium, Conservative, Aspirational, and Budget-conscious customer groups

### 📝 Key Insights
1. **Premium Segment**: High income + High spending → VIP programs and luxury products
2. **Conservative Segment**: High income + Low spending → Value-focused marketing
3. **Aspirational Segment**: Low income + High spending → Payment plans and affordable luxury
4. **Budget Segment**: Low income + Low spending → Price competition and discounts

### 🎓 Learning Outcomes
- Customer segmentation using clustering
- K-Means algorithm implementation
- Optimal cluster selection techniques
- DBSCAN clustering comparison
- Business insight generation from clusters
- Marketing strategy development based on segments

---

### 🛠️ Technologies Used
- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning library

---

## 👨‍💻 Author

**Abdelkader**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abdelkader7151)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdelrhman-abdelkader-6313a4291/)

---

*This project is part of the ElevvoPathways Machine Learning curriculum.*