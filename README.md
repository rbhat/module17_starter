# Bank Marketing Campaign Classification

A machine learning project comparing multiple classification algorithms to predict bank term deposit subscriptions from telemarketing campaign data.

## Business Objective

Predict whether a client will subscribe to a term deposit based on marketing campaign data. By accurately identifying high-potential customers, banks can focus telemarketing efforts more efficiently, reducing wasted calls and improving conversion rates.

## Results Summary

### Model Recommendation: **Logistic Regression**

After extensive testing and hyperparameter optimization, **Logistic Regression** emerges as the optimal choice for this classification task.

**Key Performance Metrics:**
- **Test Accuracy**: 89.6% (after feature engineering)
- **Training Time**: 0.028 seconds
- **Prediction Speed**: Near-instantaneous
- **Improvement over baseline**: +2.3%

### Why Logistic Regression?

1. **Best Balance**: Achieves the highest accuracy while maintaining extremely fast training and prediction times
2. **Interpretability**: Provides clear insights into which features drive term deposit subscriptions
3. **Production-Ready**: Low computational requirements make it ideal for real-time scoring
4. **Robust Performance**: Consistent results across different feature sets and data samples

## Project Overview

This analysis compares four classification algorithms on a dataset of 30,488 Portuguese bank marketing records (after data cleaning):

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)

The dataset includes 20 features covering client demographics, campaign interactions, and economic indicators.

## Key Findings

### 1. Model Performance Comparison

| Model | Test Accuracy | Training Time | Ranking Score |
|-------|--------------|---------------|---------------|
| Logistic Regression | 89.6% | 0.028s | 1.0000 |
| SVM | 89.4% | 3.824s | 0.9680 |
| KNN | 87.9% | 0.003s | 0.5507 |
| Decision Tree | 87.4% | 0.096s | 0.5289 |

*Ranking based on weighted score: 60% accuracy, 30% test speed, 10% training speed*

### 2. Feature Engineering Impact

Removing economic indicators (employment rate, consumer price index, confidence index) surprisingly **improved** model performance:
- Logistic Regression: +2.3% accuracy improvement
- SVM: +2.0% improvement  
- Decision Tree: +2.1% improvement
- KNN: +1.7% improvement

This suggests that client-specific and campaign-related features are more predictive than macroeconomic factors.

### 3. Hyperparameter Optimization Results

GridSearchCV and RandomizedSearchCV yielded modest improvements:
- KNN: +1.05% with 15 neighbors, euclidean distance
- Decision Tree: +2.08% with max_depth=3, gini criterion
- SVM: No improvement (already optimal with linear kernel)

RandomizedSearchCV proved 1.2x faster than GridSearchCV while achieving comparable results.

## Technical Implementation

### Data Processing
- Cleaned dataset by removing 10,700 rows (26%) containing 'unknown' values
- One-hot encoded categorical variables (28 features after encoding)
- Standardized features for distance-based algorithms (KNN, SVM)
- 80/20 train-test split with stratification

### Model Architecture
- Modular design with separate `models.py` for ML algorithms
- Comprehensive visualization suite in `visualization.py`
- Jupyter notebook (`prompt_III.ipynb`) for interactive analysis

### Key Libraries
- scikit-learn for machine learning algorithms
- pandas for data manipulation
- matplotlib/seaborn for visualizations
- numpy for numerical operations

## Recommendations for Production

1. **Deploy Logistic Regression** as the primary model for its optimal accuracy-speed tradeoff
2. **Focus on data quality** - removing 'unknown' values significantly improved all models
3. **Prioritize client and campaign features** over economic indicators
4. **Consider ensemble approach** combining Logistic Regression and SVM for critical decisions
5. **Implement A/B testing** to validate model predictions against actual campaign results

## Future Enhancements

1. **Feature Engineering**: Create interaction terms between client demographics and campaign timing
2. **Ensemble Methods**: Test Random Forest and Gradient Boosting for potential improvements
3. **Class Imbalance**: Explore SMOTE or class weights (87.3% negative class)
4. **Real-time Scoring**: Build API endpoint for live predictions during calls
5. **Explainability**: Add SHAP values to understand individual predictions

## Getting Started

### Prerequisites
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Analysis
```bash
jupyter notebook prompt_III.ipynb
```

### Project Structure
```
├── data/
│   ├── bank-additional-full.csv    # Full dataset (41,188 records)
│   ├── bank-additional.csv         # 10% sample for testing
│   └── bank-additional-names.txt   # Feature descriptions
├── models.py                       # ML model implementations
├── visualization.py                # Plotting utilities
├── prompt_III.ipynb               # Main analysis notebook
└── requirements.txt               # Python dependencies
```

## Citation

Dataset: [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
Moro et al., 2014. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems.

---

*This analysis demonstrates that sometimes simpler models (Logistic Regression) outperform complex alternatives, especially when considering real-world constraints like training time and interpretability.*