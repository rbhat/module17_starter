"""
Machine Learning Models for Bank Marketing Classification

This module contains functions to train and evaluate various classification models
including Logistic Regression, K-Nearest Neighbors, Decision Trees, and Support Vector Machines.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform


def train_logistic_regression(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """
    Train and evaluate Logistic Regression model
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature matrices
    y_train, y_test : array-like
        Training and test target vectors
    X_train_scaled, X_test_scaled : array-like
        Scaled versions of feature matrices
        
    Returns:
    --------
    tuple : (train_time, train_accuracy, test_time, test_accuracy)
    """
    # Create and train logistic regression model
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    
    # Measure training time
    start_time = time.time()
    log_reg.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred_train = log_reg.predict(X_train_scaled)
    start_time = time.time()
    y_pred_test = log_reg.predict(X_test_scaled)
    test_time = time.time() - start_time
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    return train_time, train_accuracy, test_time, test_accuracy


def train_knn(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, n_neighbors=5):
    """
    Train and evaluate K-Nearest Neighbors model
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature matrices
    y_train, y_test : array-like
        Training and test target vectors
    X_train_scaled, X_test_scaled : array-like
        Scaled versions of feature matrices
    n_neighbors : int, default=5
        Number of neighbors to use
        
    Returns:
    --------
    tuple : (train_time, train_accuracy, test_time, test_accuracy)
    """
    # Create and train KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Measure training time
    start_time = time.time()
    knn.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred_train = knn.predict(X_train_scaled)
    start_time = time.time()
    y_pred_test = knn.predict(X_test_scaled)
    test_time = time.time() - start_time
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    return train_time, train_accuracy, test_time, test_accuracy


def train_decision_tree(X_train, X_test, y_train, y_test, **kwargs):
    """
    Train and evaluate Decision Tree model
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature matrices
    y_train, y_test : array-like
        Training and test target vectors
    **kwargs : dict
        Additional parameters for DecisionTreeClassifier
        
    Returns:
    --------
    tuple : (train_time, train_accuracy, test_time, test_accuracy)
    """
    # Create and train Decision Tree model
    dt = DecisionTreeClassifier(random_state=42, **kwargs)
    
    # Measure training time
    start_time = time.time()
    dt.fit(X_train, y_train)  # Decision trees don't require scaled features
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred_train = dt.predict(X_train)
    start_time = time.time()
    y_pred_test = dt.predict(X_test)
    test_time = time.time() - start_time
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    return train_time, train_accuracy, test_time, test_accuracy


def train_svm(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, **kwargs):
    """
    Train and evaluate Support Vector Machine model
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature matrices
    y_train, y_test : array-like
        Training and test target vectors
    X_train_scaled, X_test_scaled : array-like
        Scaled versions of feature matrices
    **kwargs : dict
        Additional parameters for SVC
        
    Returns:
    --------
    tuple : (train_time, train_accuracy, test_time, test_accuracy)
    """
    # Create and train SVM model
    svm = SVC(random_state=42, **kwargs)
    
    # Measure training time
    start_time = time.time()
    svm.fit(X_train_scaled, y_train)  # SVM requires scaled features
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred_train = svm.predict(X_train_scaled)
    start_time = time.time()
    y_pred_test = svm.predict(X_test_scaled)
    test_time = time.time() - start_time
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    return train_time, train_accuracy, test_time, test_accuracy


def train_baseline_model(X_train, X_test, y_train, y_test, strategy='most_frequent'):
    """
    Train and evaluate a baseline model using DummyClassifier
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature matrices
    y_train, y_test : array-like
        Training and test target vectors
    strategy : str, default='most_frequent'
        Strategy to use for dummy classifier
        
    Returns:
    --------
    tuple : (baseline_accuracy, dummy_clf)
    """
    # Create a dummy classifier
    dummy_clf = DummyClassifier(strategy=strategy, random_state=42)

    # Measure training time
    start_time = time.time()    
    # Fit the dummy classifier
    dummy_clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred_dummy = dummy_clf.predict(X_test)
    
    # Calculate baseline accuracy
    baseline_accuracy = accuracy_score(y_test, y_pred_dummy)
    
    return baseline_accuracy, dummy_clf, train_time


def get_param_grids():
    """
    Get parameter grids for GridSearchCV
    
    Returns:
    --------
    dict : Parameter grids for each model
    """
    param_grids = {
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'decision_tree': {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
    return param_grids


def get_param_distributions():
    """
    Get parameter distributions for RandomizedSearchCV
    
    Returns:
    --------
    dict : Parameter distributions for each model
    """
    param_distributions = {
        'knn': {
            'n_neighbors': randint(3, 30),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': randint(1, 5)  # Parameter for minkowski distance
        },
        'decision_tree': {
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2', None]
        },
        'svm': {
            'C': uniform(0.01, 100),  # Continuous distribution from 0.01 to 100
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'] + list(uniform(0.0001, 0.1).rvs(5, random_state=42)),
            'degree': randint(2, 5)  # For poly kernel
        }
    }
    return param_distributions


def run_randomized_search(model_type, X_train, y_train, X_train_scaled=None, 
                         n_iter=30, cv=5, sample_size=None, random_state=42):
    """
    Run RandomizedSearchCV for a specific model type
    
    Parameters:
    -----------
    model_type : str
        Type of model ('knn', 'decision_tree', 'svm')
    X_train : array-like
        Training feature matrix
    y_train : array-like
        Training target vector
    X_train_scaled : array-like, optional
        Scaled training features (required for knn and svm)
    n_iter : int, default=30
        Number of parameter settings sampled
    cv : int, default=5
        Number of cross-validation folds
    sample_size : int, optional
        Sample size for training (useful for SVM)
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict : Results including best model, parameters, scores, and time
    """
    param_distributions = get_param_distributions()
    
    # Select appropriate model and features
    if model_type == 'knn':
        model = KNeighborsClassifier()
        X_fit = X_train_scaled
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=random_state)
        X_fit = X_train
    elif model_type == 'svm':
        model = SVC(random_state=random_state)
        X_fit = X_train_scaled
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use sample if specified
    if sample_size and sample_size < len(X_fit):
        sample_indices = np.random.choice(len(X_fit), size=sample_size, replace=False)
        X_fit = X_fit[sample_indices]
        y_fit = y_train.iloc[sample_indices] if hasattr(y_train, 'iloc') else y_train[sample_indices]
    else:
        y_fit = y_train
    
    # Create and run RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions[model_type],
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_fit, y_fit)
    search_time = time.time() - start_time
    
    return {
        'model': random_search,
        'best_params': random_search.best_params_,
        'best_cv_score': random_search.best_score_,
        'search_time': search_time
    }


def compare_models(models_results, baseline_accuracy):
    """
    Create a comparison DataFrame for model results
    
    Parameters:
    -----------
    models_results : dict
        Dictionary containing results for each model
    baseline_accuracy : float
        Baseline accuracy to compare against
        
    Returns:
    --------
    pandas.DataFrame : Comparison of model performances
    """
    import pandas as pd
    
    data = []
    for model_name, results in models_results.items():
        data.append({
            'Model': model_name,
            'Train Time': results['train_time'],
            'Train Accuracy': results['train_accuracy'],
            'Test Time': results['test_time'],
            'Test Accuracy': results['test_accuracy'],
            'Improvement over Baseline': results['test_accuracy'] - baseline_accuracy
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Test Accuracy', ascending=False)
    return df