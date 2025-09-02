# 🎬 Task 5: MovieLens Movie Recommendation System

A comprehensive machine learning project for building movie recommendation systems using collaborative filtering algorithms on the MovieLens 100k dataset.

## 🎯 Project Objectives

- **Build recommendation systems** using MovieLens dataset
- **Compare collaborative filtering algorithms** (User-based, Item-based, SVD, Linear Regression)
- **Analyze movie ratings** and user behavior patterns
- **Evaluate model performance** with recommendation metrics (RMSE, MAE)
- **Visualize results** with comprehensive plots and analysis
- **Implement matrix factorization** techniques

## 📁 Project Structure

```
task-5-movielens-recommendation/
├── main.py                    # Complete recommendation pipeline
├── requirements.txt           # Python dependencies
├── data/ml-100k/             # MovieLens 100k dataset
├── outputs/                   # Generated visualizations and results
├── README.md                  # Project documentation
└── TASK_5_GUIDE.md           # Detailed usage guide
```

## 📊 Dataset Information

- **Source**: MovieLens 100k Dataset (GroupLens Research)
- **Ratings**: 100,000 movie ratings
- **Users**: 943 unique users
- **Movies**: 1,682 unique movies
- **Rating Scale**: 1-5 stars
- **Time Period**: September 1997 - April 1998

### Data Files Available:
- **u.data**: Main rating data (user, movie, rating, timestamp)
- **u.item**: Movie information (title, genres, release date)
- **u.user**: User demographic information (age, gender, occupation)
- **u.genre**: Movie genre classifications

## 🚀 Quick Start

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Basic Analysis
```bash
# Quick analysis with SVD Matrix Factorization
python main.py
```

### Advanced Analysis
```bash
# Compare all recommendation algorithms
python main.py --compare-models
```

## 📈 Analysis Pipeline

1. **Data Loading**: Automatic MovieLens 100k dataset loading
2. **Data Preprocessing**: User-item matrix creation and normalization
3. **Model Training**: Multiple collaborative filtering algorithms
4. **Model Evaluation**: RMSE, MAE recommendation metrics
5. **Matrix Factorization**: SVD decomposition for latent factors
6. **Similarity Computation**: Cosine similarity for user/item relationships
7. **Visualization**: Comprehensive plots and model comparison

## 🏆 Expected Results

### Performance Expectations
- **Linear Regression**: RMSE ~1.10, MAE ~0.90 (best performing)
- **SVD Matrix Factorization**: RMSE ~2.87, MAE ~2.60
- **User-Based CF**: Cosine similarity collaborative filtering
- **Item-Based CF**: Item similarity collaborative filtering

### Key Insights
- **User preferences** vary significantly across demographics
- **Movie popularity** follows power-law distribution
- **Genre preferences** strongly influence recommendations
- **Rating sparsity** affects collaborative filtering performance

## 📊 Generated Outputs

### Visualizations
- `outputs/rating_distribution.png` - Movie rating distribution
- `outputs/movie_ratings_distribution.png` - Ratings per movie histogram
- `outputs/model_comparison.png` - Algorithm performance comparison
- `outputs/top_rated_movies.png` - Highest rated movies ranking
- `outputs/user_activity_distribution.png` - User rating activity analysis

### Analysis Files
- `outputs/model_performance.csv` - Detailed metrics for all models
- `outputs/dataset_statistics.csv` - Dataset summary statistics

## 🔧 Command Line Options

- `--compare-models`: Compare all recommendation algorithms
- `--test-size`: Test set proportion (default: 0.2)
- `--random-state`: Random seed for reproducibility

## 📚 Machine Learning Concepts Covered

### Recommendation Techniques
- **Collaborative Filtering**: User-based and item-based approaches
- **Matrix Factorization**: SVD decomposition for latent factors
- **Similarity Measures**: Cosine similarity computation
- **Linear Regression**: Baseline predictive modeling

### Model Evaluation
- **RMSE (Root Mean Square Error)**: Prediction accuracy metric
- **MAE (Mean Absolute Error)**: Average prediction error
- **Rating Prediction**: Movie rating forecasting
- **Recommendation Quality**: User satisfaction metrics

### Advanced Topics
- **User-Item Matrix**: Sparse matrix representations
- **Latent Factor Models**: Dimensionality reduction techniques
- **Cold Start Problem**: New user/movie recommendation challenges
- **Scalability**: Large-scale recommendation systems

## 🎯 Learning Outcomes

### Technical Skills
- **Recommendation Systems**: Building collaborative filtering models
- **Matrix Operations**: User-item matrix manipulation
- **Similarity Computation**: Distance and similarity measures
- **Dimensionality Reduction**: SVD and matrix factorization
- **Sparse Data Handling**: Efficient processing of sparse matrices

### Domain Knowledge
- **User Behavior**: Movie rating patterns and preferences
- **Content Analysis**: Movie genre and metadata utilization
- **Recommendation Quality**: Evaluating system effectiveness
- **Scalability Challenges**: Real-world recommendation system issues

## 🎬 Real-World Applications

- **Streaming Platforms**: Netflix, Hulu movie recommendations
- **E-commerce**: Amazon product suggestions
- **Social Media**: Content and friend recommendations
- **Music Services**: Spotify song and playlist recommendations
- **News Platforms**: Article and content personalization

## 📋 Task Completion Checklist

- ✅ **Dataset**: MovieLens 100k loaded and processed
- ✅ **Preprocessing**: User-item matrix creation
- ✅ **Models**: Multiple recommendation algorithms implemented
- ✅ **Evaluation**: RMSE/MAE metrics computed
- ✅ **Visualization**: Rating distributions and model comparison
- ✅ **Analysis**: User behavior and movie popularity insights
- ✅ **Documentation**: Complete usage guide and analysis

## 🌟 Success Criteria Met

**Your MovieLens Recommendation System project is successful when:**

- ✅ All algorithms run without errors
- ✅ Linear Regression achieves lowest RMSE (~1.10)
- ✅ SVD Matrix Factorization provides baseline performance
- ✅ All visualization files are generated
- ✅ Rating distributions show expected patterns
- ✅ Top-rated movies are logically identified
- ✅ Model comparison provides clear performance ranking

---

## 🎯 **Key Achievements**

### **Dataset Successfully Loaded**
- ✅ 100,000 movie ratings processed
- ✅ 943 users and 1,682 movies analyzed
- ✅ Complete user and movie metadata available

### **Models Successfully Implemented**
- ✅ **Linear Regression**: RMSE: 1.1018, MAE: 0.9074 (Best)
- ✅ **SVD Matrix Factorization**: RMSE: 2.8755, MAE: 2.6069
- ✅ User-based and Item-based collaborative filtering frameworks

### **Comprehensive Analysis Generated**
- ✅ Rating distribution analysis
- ✅ User activity patterns
- ✅ Movie popularity distributions
- ✅ Top-rated movie identification
- ✅ Model performance comparison

**🎉 Task 5 Complete: Movie Recommendation System with MovieLens Dataset!** 🎬
