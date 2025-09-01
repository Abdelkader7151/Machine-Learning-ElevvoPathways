# Customer Segmentation - Complete Solution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== Customer Segmentation Analysis ===")
    
    # Step 1: Load Dataset
    try:
        df = pd.read_csv('data/Mall_Customers.csv')
        print(f"Dataset loaded successfully")
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Dataset not found. Please ensure 'Mall_Customers.csv' is in the 'data/' folder")
        return
    
    # Step 2: Dataset Overview
    print("\n=== Dataset Overview ===")
    print("Columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Step 3: Data Preprocessing
    print("\n=== Data Preprocessing ===")
    print("Missing values:")
    print(df.isnull().sum())
    
    # Select features for clustering (Annual Income and Spending Score)
    # Handle different possible column names
    income_cols = ['Annual Income (k$)', 'Annual_Income_(k$)', 'Annual_Income', 'AnnualIncome']
    spending_cols = ['Spending Score (1-100)', 'Spending_Score_(1-100)', 'Spending_Score', 'SpendingScore']
    
    income_col = None
    spending_col = None
    
    for col in income_cols:
        if col in df.columns:
            income_col = col
            break
    
    for col in spending_cols:
        if col in df.columns:
            spending_col = col
            break
    
    if not income_col or not spending_col:
        print(f"Required columns not found. Available columns: {df.columns.tolist()}")
        return
    
    print(f"Using columns: {income_col}, {spending_col}")
    
    # Extract features for clustering
    X = df[[income_col, spending_col]].copy()
    
    # Step 4: Exploratory Data Analysis
    print(f"\n=== Exploratory Data Analysis ===")
    print(f"Income range: {X[income_col].min():.1f} - {X[income_col].max():.1f}")
    print(f"Spending Score range: {X[spending_col].min():.1f} - {X[spending_col].max():.1f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Distribution plots
    plt.subplot(1, 3, 1)
    plt.hist(X[income_col], bins=20, alpha=0.7, color='skyblue')
    plt.title('Distribution of Annual Income')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(X[spending_col], bins=20, alpha=0.7, color='lightcoral')
    plt.title('Distribution of Spending Score')
    plt.xlabel('Spending Score (1-100)')
    plt.ylabel('Frequency')
    
    # Scatter plot
    plt.subplot(1, 3, 3)
    plt.scatter(X[income_col], X[spending_col], alpha=0.6, color='green')
    plt.title('Income vs Spending Score')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/data_exploration.png', dpi=300, bbox_inches='tight')
    print("Data exploration plots saved to assets/data_exploration.png")
    plt.close()
    
    # Additional Demographics Analysis
    print("\n=== Demographics Analysis ===")
    
    # Age and Gender Analysis
    plt.figure(figsize=(15, 10))
    
    # Age distribution
    plt.subplot(2, 3, 1)
    plt.hist(df['Age'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # Gender distribution
    plt.subplot(2, 3, 2)
    gender_counts = df['Gender'].value_counts()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
            colors=['lightblue', 'lightpink'])
    plt.title('Gender Distribution')
    
    # Age vs Income
    plt.subplot(2, 3, 3)
    plt.scatter(df['Age'], df[income_col], alpha=0.6, color='purple')
    plt.title('Age vs Annual Income')
    plt.xlabel('Age')
    plt.ylabel('Annual Income (k$)')
    
    # Age vs Spending Score
    plt.subplot(2, 3, 4)
    plt.scatter(df['Age'], df[spending_col], alpha=0.6, color='brown')
    plt.title('Age vs Spending Score')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    
    # Gender vs Income boxplot
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='Gender', y=income_col, palette='Set2')
    plt.title('Income by Gender')
    plt.ylabel('Annual Income (k$)')
    
    # Gender vs Spending boxplot
    plt.subplot(2, 3, 6)
    sns.boxplot(data=df, x='Gender', y=spending_col, palette='Set3')
    plt.title('Spending Score by Gender')
    plt.ylabel('Spending Score (1-100)')
    
    plt.tight_layout()
    plt.savefig('assets/demographics_analysis.png', dpi=300, bbox_inches='tight')
    print("Demographics analysis plots saved to assets/demographics_analysis.png")
    plt.close()
    
    # Step 5: Feature Scaling
    print("\n=== Feature Scaling ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled using StandardScaler")
    
    # Step 6: Determine Optimal Number of Clusters
    print("\n=== Finding Optimal Number of Clusters ===")
    
    # Elbow Method
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot Elbow Method and Silhouette Analysis
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, 'ro-')
    plt.title('Silhouette Score For Different k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/cluster_optimization.png', dpi=300, bbox_inches='tight')
    print("Cluster optimization plots saved to assets/cluster_optimization.png")
    plt.close()
    
    # Find optimal k (highest silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Silhouette Score: {max(silhouette_scores):.3f}")
    
    # Step 7: K-Means Clustering
    print(f"\n=== K-Means Clustering (k={optimal_k}) ===")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to original dataframe
    df['Cluster'] = cluster_labels
    X['Cluster'] = cluster_labels
    
    print(f"K-Means clustering completed")
    print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.3f}")
    
    # Step 8: Cluster Analysis
    print(f"\n=== Cluster Analysis ===")
    
    # Cluster summary
    cluster_summary = X.groupby('Cluster').agg({
        income_col: ['count', 'mean', 'std'],
        spending_col: ['mean', 'std']
    }).round(2)
    
    print("Cluster Summary:")
    print(cluster_summary)
    
    # Individual cluster characteristics
    for i in range(optimal_k):
        cluster_data = X[X['Cluster'] == i]
        avg_income = cluster_data[income_col].mean()
        avg_spending = cluster_data[spending_col].mean()
        size = len(cluster_data)
        
        print(f"\nCluster {i}:")
        print(f"  Size: {size} customers ({size/len(X)*100:.1f}%)")
        print(f"  Average Income: ${avg_income:.1f}k")
        print(f"  Average Spending Score: {avg_spending:.1f}")
        
        # Categorize cluster
        if avg_income > X[income_col].mean() and avg_spending > X[spending_col].mean():
            category = "High Income, High Spending (Premium)"
        elif avg_income > X[income_col].mean() and avg_spending < X[spending_col].mean():
            category = "High Income, Low Spending (Conservative)"
        elif avg_income < X[income_col].mean() and avg_spending > X[spending_col].mean():
            category = "Low Income, High Spending (Aspirational)"
        else:
            category = "Low Income, Low Spending (Budget-conscious)"
        
        print(f"  Category: {category}")
    
    # Step 9: Visualization
    print(f"\n=== Cluster Visualization ===")
    
    plt.figure(figsize=(12, 5))
    
    # K-Means clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[income_col], X[spending_col], 
                         c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
    plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],
                scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title(f'K-Means Clustering (k={optimal_k})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DBSCAN clustering (bonus)
    plt.subplot(1, 2, 2)
    dbscan = DBSCAN(eps=0.6, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    scatter2 = plt.scatter(X[income_col], X[spending_col], 
                          c=dbscan_labels, cmap='plasma', alpha=0.7, s=50)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.colorbar(scatter2, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    print(f"DBSCAN found {n_clusters_dbscan} clusters and {n_noise} noise points")
    
    plt.tight_layout()
    plt.savefig('assets/clustering_results.png', dpi=300, bbox_inches='tight')
    print("Clustering results saved to assets/clustering_results.png")
    plt.close()
    
    # Additional Cluster Analysis Visualizations
    print("\n=== Advanced Cluster Visualizations ===")
    
    # Cluster characteristics analysis
    plt.figure(figsize=(20, 12))
    
    # 1. Cluster size distribution
    plt.subplot(3, 4, 1)
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    plt.bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', alpha=0.7)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    for i, v in enumerate(cluster_sizes.values):
        plt.text(i, v + 1, str(v), ha='center', va='bottom')
    
    # 2. Average income by cluster
    plt.subplot(3, 4, 2)
    avg_income = df.groupby('Cluster')[income_col].mean()
    plt.bar(avg_income.index, avg_income.values, color='lightgreen', alpha=0.7)
    plt.title('Average Income by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Income (k$)')
    for i, v in enumerate(avg_income.values):
        plt.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    # 3. Average spending by cluster
    plt.subplot(3, 4, 3)
    avg_spending = df.groupby('Cluster')[spending_col].mean()
    plt.bar(avg_spending.index, avg_spending.values, color='lightcoral', alpha=0.7)
    plt.title('Average Spending by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Spending Score')
    for i, v in enumerate(avg_spending.values):
        plt.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    # 4. Age distribution by cluster
    plt.subplot(3, 4, 4)
    avg_age = df.groupby('Cluster')['Age'].mean()
    plt.bar(avg_age.index, avg_age.values, color='orange', alpha=0.7)
    plt.title('Average Age by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Age')
    for i, v in enumerate(avg_age.values):
        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    # 5. Gender distribution by cluster
    plt.subplot(3, 4, 5)
    gender_cluster = pd.crosstab(df['Cluster'], df['Gender'], normalize='index') * 100
    gender_cluster.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightpink'])
    plt.title('Gender Distribution by Cluster (%)')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage')
    plt.legend(title='Gender')
    plt.xticks(rotation=0)
    
    # 6. 3D scatter plot (Age, Income, Spending)
    ax = plt.subplot(3, 4, 6, projection='3d')
    scatter = ax.scatter(df['Age'], df[income_col], df[spending_col], 
                        c=df['Cluster'], cmap='viridis', alpha=0.6)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score')
    ax.set_title('3D Cluster Visualization')
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    # 7. Income vs Spending with Age color coding
    plt.subplot(3, 4, 7)
    scatter = plt.scatter(df[income_col], df[spending_col], 
                         c=df['Age'], cmap='coolwarm', alpha=0.6, s=50)
    plt.title('Income vs Spending (Age Color-coded)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.colorbar(scatter, label='Age')
    
    # 8. Cluster silhouette analysis
    plt.subplot(3, 4, 8)
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
    df['Silhouette'] = silhouette_vals
    avg_silhouette = df.groupby('Cluster')['Silhouette'].mean()
    plt.bar(avg_silhouette.index, avg_silhouette.values, color='gold', alpha=0.7)
    plt.title('Average Silhouette Score by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Silhouette Score')
    plt.axhline(y=silhouette_score(X_scaled, cluster_labels), color='red', 
                linestyle='--', label=f'Overall: {silhouette_score(X_scaled, cluster_labels):.3f}')
    plt.legend()
    
    # 9. Violin plot - Income distribution by cluster
    plt.subplot(3, 4, 9)
    sns.violinplot(data=df, x='Cluster', y=income_col, palette='Set1')
    plt.title('Income Distribution by Cluster')
    plt.ylabel('Annual Income (k$)')
    
    # 10. Violin plot - Spending distribution by cluster
    plt.subplot(3, 4, 10)
    sns.violinplot(data=df, x='Cluster', y=spending_col, palette='Set2')
    plt.title('Spending Distribution by Cluster')
    plt.ylabel('Spending Score (1-100)')
    
    # 11. Heatmap of cluster characteristics
    plt.subplot(3, 4, 11)
    cluster_stats = df.groupby('Cluster')[['Age', income_col, spending_col]].mean()
    # Normalize for better visualization
    cluster_stats_norm = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())
    sns.heatmap(cluster_stats_norm.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Normalized Values'})
    plt.title('Normalized Cluster Characteristics')
    plt.xlabel('Cluster')
    
    # 12. Customer value analysis (Income * Spending proxy)
    plt.subplot(3, 4, 12)
    df['Customer_Value'] = df[income_col] * df[spending_col] / 100
    avg_value = df.groupby('Cluster')['Customer_Value'].mean()
    plt.bar(avg_value.index, avg_value.values, color='purple', alpha=0.7)
    plt.title('Average Customer Value by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Customer Value Index')
    for i, v in enumerate(avg_value.values):
        plt.text(i, v + 5, f'{v:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('assets/advanced_cluster_analysis.png', dpi=300, bbox_inches='tight')
    print("Advanced cluster analysis plots saved to assets/advanced_cluster_analysis.png")
    plt.close()
    
    # Detailed comparison plots
    plt.figure(figsize=(16, 8))
    
    # Comparison of different clustering algorithms
    plt.subplot(2, 3, 1)
    plt.scatter(X[income_col], X[spending_col], c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
               c='red', marker='x', s=200, linewidths=3)
    plt.title(f'K-Means (k={optimal_k})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    
    plt.subplot(2, 3, 2)
    plt.scatter(X[income_col], X[spending_col], c=dbscan_labels, cmap='plasma', alpha=0.7, s=50)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    
    # Try different K values for comparison
    plt.subplot(2, 3, 3)
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_3 = kmeans_3.fit_predict(X_scaled)
    plt.scatter(X[income_col], X[spending_col], c=labels_3, cmap='Set1', alpha=0.7, s=50)
    centroids_3 = scaler.inverse_transform(kmeans_3.cluster_centers_)
    plt.scatter(centroids_3[:, 0], centroids_3[:, 1],
               c='red', marker='x', s=200, linewidths=3)
    plt.title('K-Means (k=3)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    
    plt.subplot(2, 3, 4)
    kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_4 = kmeans_4.fit_predict(X_scaled)
    plt.scatter(X[income_col], X[spending_col], c=labels_4, cmap='Set2', alpha=0.7, s=50)
    centroids_4 = scaler.inverse_transform(kmeans_4.cluster_centers_)
    plt.scatter(centroids_4[:, 0], centroids_4[:, 1],
               c='red', marker='x', s=200, linewidths=3)
    plt.title('K-Means (k=4)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    
    plt.subplot(2, 3, 5)
    kmeans_6 = KMeans(n_clusters=6, random_state=42, n_init=10)
    labels_6 = kmeans_6.fit_predict(X_scaled)
    plt.scatter(X[income_col], X[spending_col], c=labels_6, cmap='tab10', alpha=0.7, s=50)
    centroids_6 = scaler.inverse_transform(kmeans_6.cluster_centers_)
    plt.scatter(centroids_6[:, 0], centroids_6[:, 1],
               c='red', marker='x', s=200, linewidths=3)
    plt.title('K-Means (k=6)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    
    # Silhouette comparison
    plt.subplot(2, 3, 6)
    k_values = [3, 4, optimal_k, 6]
    sil_scores = [
        silhouette_score(X_scaled, labels_3),
        silhouette_score(X_scaled, labels_4),
        silhouette_score(X_scaled, cluster_labels),
        silhouette_score(X_scaled, labels_6)
    ]
    plt.bar(k_values, sil_scores, color='gold', alpha=0.7)
    plt.title('Silhouette Score Comparison')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.axhline(y=max(sil_scores), color='red', linestyle='--', alpha=0.5)
    for i, v in enumerate(sil_scores):
        plt.text(k_values[i], v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('assets/clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("Clustering comparison plots saved to assets/clustering_comparison.png")
    plt.close()
    
    # Step 10: Business Insights
    print(f"\n=== Business Insights ===")
    print("Customer Segmentation Recommendations:")
    
    for i in range(optimal_k):
        cluster_data = X[X['Cluster'] == i]
        avg_income = cluster_data[income_col].mean()
        avg_spending = cluster_data[spending_col].mean()
        size = len(cluster_data)
        
        print(f"\nCluster {i} Strategy:")
        if avg_income > X[income_col].mean() and avg_spending > X[spending_col].mean():
            print("  - Target with premium products and exclusive offers")
            print("  - Focus on luxury and quality messaging")
            print("  - Implement VIP programs and personalized service")
        elif avg_income > X[income_col].mean() and avg_spending < X[spending_col].mean():
            print("  - Emphasize value and quality over price")
            print("  - Offer rational, benefit-focused marketing")
            print("  - Provide detailed product information and comparisons")
        elif avg_income < X[income_col].mean() and avg_spending > X[spending_col].mean():
            print("  - Offer attractive payment plans and financing")
            print("  - Focus on aspirational and lifestyle marketing")
            print("  - Provide affordable luxury alternatives")
        else:
            print("  - Compete on price and value")
            print("  - Offer discounts and budget-friendly options")
            print("  - Focus on essential products and basic needs")
    
    print(f"\n=== Analysis Complete! ===")
    print(f"Generated visualizations:")
    print(f"  - assets/data_exploration.png - Basic data exploration")
    print(f"  - assets/demographics_analysis.png - Age and gender analysis")
    print(f"  - assets/cluster_optimization.png - Elbow method and silhouette analysis")
    print(f"  - assets/clustering_results.png - K-Means and DBSCAN results")
    print(f"  - assets/advanced_cluster_analysis.png - Comprehensive cluster characteristics")
    print(f"  - assets/clustering_comparison.png - Different K values and algorithm comparison")

if __name__ == "__main__":
    main()