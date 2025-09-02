#!/usr/bin/env python3
"""
Task 5: MovieLens Movie Recommendation System
============================================

A comprehensive machine learning project for building movie recommendation systems
using collaborative filtering algorithms on the MovieLens 100k dataset.

Author: AI Assistant
Date: 2024
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MovieRecommender:
    """
    Movie Recommendation System using Multiple Collaborative Filtering Algorithms
    """

    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the movie recommender

        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None

        # Create outputs directory
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)

        print("🎬 MovieLens Movie Recommendation System")
        print("=" * 50)

    def load_data(self):
        """Load and prepare the MovieLens 100k dataset"""
        print("\n📊 Loading MovieLens 100k Dataset...")

        # Load the ratings dataset
        ratings_path = Path("data/ml-100k/u.data")
        if not ratings_path.exists():
            print("❌ MovieLens dataset not found. Please download it first.")
            return None

        self.ratings_df = pd.read_csv(ratings_path, sep='\t',
                                     names=['user_id', 'movie_id', 'rating', 'timestamp'])

        # Load additional data files
        self.movies_df = self.load_movies_data()
        self.users_df = self.load_users_data()

        print(f"✅ Dataset loaded: {len(self.ratings_df)} ratings")
        print(f"📈 Users: {len(self.ratings_df['user_id'].unique())}")
        print(f"🎬 Movies: {len(self.ratings_df['movie_id'].unique())}")
        print(f"⭐ Rating scale: {self.ratings_df['rating'].min()}-{self.ratings_df['rating'].max()}")

        # Split the data
        train_data, test_data = train_test_split(
            self.ratings_df, test_size=self.test_size, random_state=self.random_state
        )

        print(f"🔀 Train/Test split: {len(train_data)}/{len(test_data)} ratings")

        return train_data, test_data

    def load_movies_data(self):
        """Load movie information"""
        movies_path = Path("data/ml-100k/u.item")
        if movies_path.exists():
            movies_df = pd.read_csv(movies_path, sep='|', encoding='latin-1',
                                   names=['movie_id', 'movie_title', 'release_date', 'video_release_date',
                                          'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                          'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                          'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi',
                                          'Thriller', 'War', 'Western'])
            return movies_df
        return None

    def load_users_data(self):
        """Load user information"""
        users_path = Path("data/ml-100k/u.user")
        if users_path.exists():
            users_df = pd.read_csv(users_path, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
            return users_df
        return None

    def initialize_models(self):
        """Initialize all recommendation models"""
        print("\n🤖 Initializing recommendation models...")

        self.models = {
            'User-Based CF': 'user_based',
            'Item-Based CF': 'item_based',
            'SVD Matrix Factorization': 'svd',
            'Linear Regression': 'linear'
        }

        print(f"✅ Initialized {len(self.models)} recommendation models")

    def train_and_evaluate(self, train_data, test_data, compare_models=False):
        """Train and evaluate all models"""
        print("\n🎯 Training and evaluating models...")

        for name, model_type in self.models.items():
            print(f"\n🔄 Training {name}...")

            try:
                if model_type == 'user_based':
                    rmse, mae = self.user_based_cf(train_data, test_data)
                elif model_type == 'item_based':
                    rmse, mae = self.item_based_cf(train_data, test_data)
                elif model_type == 'svd':
                    rmse, mae = self.svd_recommendation(train_data, test_data)
                elif model_type == 'linear':
                    rmse, mae = self.linear_regression_model(train_data, test_data)

                # Store results
                self.results[name] = {
                    'rmse': rmse,
                    'mae': mae
                }

                print(".4f"                      ".4f")

                if not compare_models and name == 'SVD Matrix Factorization':
                    break

            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue

    def user_based_cf(self, train_data, test_data):
        """User-based collaborative filtering"""
        # Create user-item matrix
        user_item_matrix = train_data.pivot_table(
            index='user_id', columns='movie_id', values='rating'
        ).fillna(0)

        # Calculate user similarities
        user_similarity = self.cosine_similarity(user_item_matrix)

        # Make predictions
        predictions = []
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            actual_rating = row['rating']

            if user_id in user_similarity.index and movie_id in user_item_matrix.columns:
                # Find similar users who rated this movie
                similar_users = user_similarity.loc[user_id]
                movie_ratings = user_item_matrix[movie_id]

                # Weighted average prediction
                mask = movie_ratings > 0
                if mask.sum() > 0:
                    weights = similar_users[mask]
                    ratings = movie_ratings[mask]
                    pred_rating = np.average(ratings, weights=weights)
                else:
                    pred_rating = user_item_matrix.mean().mean()
            else:
                pred_rating = user_item_matrix.mean().mean()

            predictions.append(pred_rating)

        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)

        return rmse, mae

    def item_based_cf(self, train_data, test_data):
        """Item-based collaborative filtering"""
        # Create item-user matrix
        item_user_matrix = train_data.pivot_table(
            index='movie_id', columns='user_id', values='rating'
        ).fillna(0)

        # Calculate item similarities
        item_similarity = self.cosine_similarity(item_user_matrix)

        # Make predictions
        predictions = []
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            actual_rating = row['rating']

            if movie_id in item_similarity.index and user_id in item_user_matrix.columns:
                # Find similar movies rated by this user
                similar_items = item_similarity.loc[movie_id]
                user_ratings = item_user_matrix[user_id]

                # Weighted average prediction
                mask = user_ratings > 0
                if mask.sum() > 0:
                    weights = similar_items[mask]
                    ratings = user_ratings[mask]
                    pred_rating = np.average(ratings, weights=weights)
                else:
                    pred_rating = item_user_matrix.mean().mean()
            else:
                pred_rating = item_user_matrix.mean().mean()

            predictions.append(pred_rating)

        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)

        return rmse, mae

    def svd_recommendation(self, train_data, test_data):
        """SVD-based matrix factorization"""
        # Create user-item matrix
        user_item_matrix = train_data.pivot_table(
            index='user_id', columns='movie_id', values='rating'
        ).fillna(0)

        # Apply SVD
        svd = TruncatedSVD(n_components=min(50, len(user_item_matrix.columns)-1),
                          random_state=self.random_state)
        user_factors = svd.fit_transform(user_item_matrix)
        item_factors = svd.components_.T

        # Reconstruct matrix
        reconstructed = np.dot(user_factors, item_factors.T)

        # Make predictions
        predictions = []
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']

            if user_id in user_item_matrix.index and movie_id in user_item_matrix.columns:
                user_idx = user_item_matrix.index.get_loc(user_id)
                movie_idx = user_item_matrix.columns.get_loc(movie_id)
                pred_rating = reconstructed[user_idx, movie_idx]
            else:
                pred_rating = user_item_matrix.mean().mean()

            predictions.append(pred_rating)

        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)

        return rmse, mae

    def linear_regression_model(self, train_data, test_data):
        """Simple linear regression model"""
        # Prepare features
        train_features = train_data[['user_id', 'movie_id']].values
        train_target = train_data['rating'].values

        test_features = test_data[['user_id', 'movie_id']].values
        test_target = test_data['rating'].values

        # Train linear regression
        lr = LinearRegression()
        lr.fit(train_features, train_target)

        # Make predictions
        predictions = lr.predict(test_features)

        # Clip predictions to valid rating range
        predictions = np.clip(predictions, 1, 5)

        rmse = np.sqrt(mean_squared_error(test_target, predictions))
        mae = mean_absolute_error(test_target, predictions)

        return rmse, mae

    def cosine_similarity(self, matrix):
        """Calculate cosine similarity between rows"""
        # Convert to numpy array for calculations
        matrix_np = matrix.values

        # Normalize matrix
        norms = np.sqrt(np.sum(matrix_np**2, axis=1, keepdims=True))
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_matrix = matrix_np / norms

        # Calculate cosine similarity
        similarity = np.dot(normalized_matrix, normalized_matrix.T)

        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            similarity,
            index=matrix.index,
            columns=matrix.index
        )

        return similarity_df

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating visualizations...")

        # 1. Rating distribution
        plt.figure(figsize=(10, 6))
        ratings = self.ratings_df['rating']
        plt.hist(ratings, bins=5, alpha=0.7, edgecolor='black')
        plt.xlabel('Movie Rating')
        plt.ylabel('Frequency')
        plt.title('Distribution of Movie Ratings')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.outputs_dir / 'rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Number of ratings per movie
        movie_ratings_count = self.ratings_df.groupby('movie_id')['rating'].count()
        plt.figure(figsize=(12, 6))
        plt.hist(movie_ratings_count, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Ratings per Movie')
        plt.ylabel('Frequency')
        plt.title('Distribution of Ratings per Movie')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.outputs_dir / 'movie_ratings_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Model comparison
        if len(self.results) > 1:
            plt.figure(figsize=(14, 8))

            model_names = list(self.results.keys())
            rmse_scores = [self.results[name]['rmse'] for name in model_names]
            mae_scores = [self.results[name]['mae'] for name in model_names]

            x = np.arange(len(model_names))
            width = 0.35

            plt.bar(x - width/2, rmse_scores, width, label='RMSE', alpha=0.8)
            plt.bar(x + width/2, mae_scores, width, label='MAE', alpha=0.8)

            plt.xlabel('Models')
            plt.ylabel('Error Score')
            plt.title('Model Performance Comparison (Lower is Better)')
            plt.xticks(x, model_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.outputs_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Top rated movies
        if self.movies_df is not None:
            movie_stats = self.ratings_df.groupby('movie_id').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            movie_stats.columns = ['movie_id', 'rating_count', 'avg_rating']

            # Filter movies with at least 50 ratings
            popular_movies = movie_stats[movie_stats['rating_count'] >= 50]
            top_movies = popular_movies.nlargest(10, 'avg_rating')

            # Merge with movie titles
            top_movies_with_titles = top_movies.merge(
                self.movies_df[['movie_id', 'movie_title']],
                on='movie_id', how='left'
            )

            plt.figure(figsize=(12, 8))
            bars = plt.barh(top_movies_with_titles['movie_title'].head(10),
                           top_movies_with_titles['avg_rating'].head(10))
            plt.xlabel('Average Rating')
            plt.ylabel('Movie Title')
            plt.title('Top 10 Highest Rated Movies (50+ ratings)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.outputs_dir / 'top_rated_movies.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 5. User activity analysis
        if self.users_df is not None:
            user_activity = self.ratings_df.groupby('user_id')['rating'].count()
            plt.figure(figsize=(12, 6))
            plt.hist(user_activity, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Number of Ratings per User')
            plt.ylabel('Frequency')
            plt.title('User Activity Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.outputs_dir / 'user_activity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("✅ Visualizations saved to outputs/ directory")

    def save_results(self):
        """Save model results and analysis to files"""
        print("\n💾 Saving results...")

        # Save detailed results
        results_summary = []
        for name, result in self.results.items():
            results_summary.append({
                'Model': name,
                'RMSE': result['rmse'],
                'MAE': result['mae']
            })

        results_df = pd.DataFrame(results_summary)
        results_df.to_csv(self.outputs_dir / 'model_performance.csv', index=False)

        # Save best model
        if self.results:
            best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
            print(f"✅ Best model: {best_model_name}")

        # Save dataset statistics
        if self.ratings_df is not None:
            dataset_stats = {
                'total_ratings': len(self.ratings_df),
                'unique_users': len(self.ratings_df['user_id'].unique()),
                'unique_movies': len(self.ratings_df['movie_id'].unique()),
                'rating_scale': f"{self.ratings_df['rating'].min()}-{self.ratings_df['rating'].max()}",
                'avg_rating': self.ratings_df['rating'].mean(),
                'sparsity': 1 - (len(self.ratings_df) / (len(self.ratings_df['user_id'].unique()) * len(self.ratings_df['movie_id'].unique())))
            }

            stats_df = pd.DataFrame([dataset_stats])
            stats_df.to_csv(self.outputs_dir / 'dataset_statistics.csv', index=False)

        print("✅ Results saved to outputs/ directory")

    def print_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("📊 MOVIELENS RECOMMENDATION SYSTEM - RESULTS SUMMARY")
        print("="*60)

        print("\n🎬 Dataset Information:")
        if self.ratings_df is not None:
            print(f"   • Ratings: {len(self.ratings_df)} total ratings")
            print(f"   • Users: {len(self.ratings_df['user_id'].unique())}")
            print(f"   • Movies: {len(self.ratings_df['movie_id'].unique())}")
            print(f"   • Rating Scale: {self.ratings_df['rating'].min()}-{self.ratings_df['rating'].max()}")
            sparsity = 1 - (len(self.ratings_df) / (len(self.ratings_df['user_id'].unique()) * len(self.ratings_df['movie_id'].unique())))
            print(".4f")

        print("\n📈 Model Performance:")
        for name, result in sorted(self.results.items(),
                                  key=lambda x: x[1]['rmse']):
            print(f"   • {name}:")
            print(f"     - RMSE: {result['rmse']:.4f}")
            print(f"     - MAE: {result['mae']:.4f}")

        if self.results:
            best_model = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
            print(f"\n🏆 Best Model: {best_model}")
            print(".4f")

        print("\n📁 Generated Files:")
        output_files = [
            'rating_distribution.png',
            'movie_ratings_distribution.png',
            'model_comparison.png',
            'top_rated_movies.png',
            'user_activity_distribution.png',
            'model_performance.csv',
            'dataset_statistics.csv'
        ]

        for file in output_files:
            if (self.outputs_dir / file).exists():
                print(f"   ✅ {file}")

        print("\n🎯 Analysis Complete!")
        print("="*60)


def main():
    """Main function to run the movie recommendation analysis"""
    print("🚀 Starting MovieLens Recommendation System...")
    parser = argparse.ArgumentParser(description='MovieLens Movie Recommendation System')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare multiple recommendation algorithms')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()
    print("✅ Arguments parsed successfully")

    # Initialize recommender
    print("🔧 Initializing recommender...")
    recommender = MovieRecommender(
        test_size=args.test_size,
        random_state=args.random_state
    )
    print("✅ Recommender initialized")

    # Load data
    train_data, test_data = recommender.load_data()
    if train_data is None:
        print("❌ Failed to load dataset. Please ensure MovieLens dataset is properly downloaded.")
        return

    # Initialize models
    recommender.initialize_models()

    # Train and evaluate models
    recommender.train_and_evaluate(train_data, test_data, compare_models=args.compare_models)

    # Create visualizations
    recommender.create_visualizations()

    # Save results
    recommender.save_results()

    # Print summary
    recommender.print_summary()


if __name__ == "__main__":
    main()
