# Machine Learning - ElevvoPathways

A collection of machine learning projects and tasks for skill development.

## 📊 Task 1: Student Performance Indicator

Predicting student exam scores using various performance factors.

### 🎯 Project Overview
- **Dataset**: Student performance factors including study hours, attendance, parental involvement, etc.
- **Goal**: Build a regression model to predict exam scores
- **Approach**: Linear regression with feature analysis and visualization

### 📁 Project Structure
```
Task-1-student-performance-indicator/
├── data/
│   └── StudentPerformanceFactors.csv    # Dataset
├── assets/
│   ├── analysis_overview.png            # EDA visualizations
│   ├── model_predictions.png            # Model performance plots
│   ├── plot1.png                        # Additional plots
│   └── plot2.png                        # Additional plots
├── main.py                              # Complete Python script
└── Task-1-student-performance-indicator.ipynb  # Jupyter notebook
```

### 🚀 How to Run

#### Option 1: Python Script
```bash
cd Task-1-student-performance-indicator
python main.py
```

#### Option 2: Jupyter Notebook
Open `Task-1-student-performance-indicator.ipynb` in Jupyter Lab/Notebook and run all cells.

### 📈 Results & Visualizations

#### Data Analysis Overview
![Analysis Overview](Task-1-student-performance-indicator/assets/analysis_overview.png)

*Shows score distribution, study hours vs scores relationship, and correlation matrix*

#### Model Performance
![Model Predictions](Task-1-student-performance-indicator/assets/model_predictions.png)

*Actual vs Predicted scores and residuals distribution*

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

*This project is part of the ElevvoPathways Machine Learning curriculum.*
