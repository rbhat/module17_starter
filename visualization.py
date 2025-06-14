"""
Visualization utilities for Bank Marketing Classification

This module contains functions for creating plots and visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_model_comparison(comparison_df, title="Model Performance Comparison"):
    """
    Create a bar plot comparing model performances
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        DataFrame with model comparison results
    title : str
        Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot test accuracy
    models = comparison_df['Model']
    test_acc = comparison_df['Test Accuracy']
    
    ax1.bar(models, test_acc)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy by Model')
    ax1.set_ylim(0.8, max(test_acc) * 1.02)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot training time
    train_time = comparison_df['Train Time']
    
    ax2.bar(models, train_time)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time by Model')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_hyperparameter_importance(grid_search_results, model_name):
    """
    Plot hyperparameter importance from grid search results
    
    Parameters:
    -----------
    grid_search_results : GridSearchCV or RandomizedSearchCV object
        Fitted search object
    model_name : str
        Name of the model for the title
    """
    results_df = pd.DataFrame(grid_search_results.cv_results_)
    
    # Get parameter names
    param_names = [key for key in results_df.columns if key.startswith('param_')]
    
    # Calculate parameter importance (variance in scores for each parameter value)
    importances = {}
    
    for param in param_names:
        param_clean = param.replace('param_', '')
        unique_values = results_df[param].unique()
        
        scores_by_value = []
        for value in unique_values:
            mask = results_df[param] == value
            scores = results_df.loc[mask, 'mean_test_score'].values
            scores_by_value.append(scores.mean())
        
        importances[param_clean] = np.std(scores_by_value)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    params = list(importances.keys())
    values = list(importances.values())
    
    ax.barh(params, values)
    ax.set_xlabel('Importance (std of mean scores)')
    ax.set_title(f'Hyperparameter Importance for {model_name}')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix for classification results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for the title
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()


def plot_learning_curves(estimator, X, y, cv=5, n_jobs=-1, 
                        train_sizes=np.linspace(0.1, 1.0, 10),
                        title="Learning Curves"):
    """
    Plot learning curves for an estimator
    
    Parameters:
    -----------
    estimator : sklearn estimator
        The model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : int, default=5
        Number of cross-validation folds
    n_jobs : int, default=-1
        Number of parallel jobs
    train_sizes : array-like
        Training set sizes to evaluate
    title : str
        Title for the plot
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def plot_roc_curves(models_dict, X_test, y_test, X_test_scaled=None):
    """
    Plot ROC curves for multiple models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: fitted_model}
    X_test : array-like
        Test feature matrix
    y_test : array-like
        Test target vector
    X_test_scaled : array-like, optional
        Scaled test features for models that need it
    """
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        # Determine which features to use
        if name in ['SVM', 'KNN', 'Logistic Regression'] and X_test_scaled is not None:
            X_use = X_test_scaled
        else:
            X_use = X_test
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_use)[:, 1]
        else:
            # For SVM, use decision function
            y_score = model.decision_function(X_use)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def create_performance_report(models_results, baseline_accuracy, save_path=None):
    """
    Create a comprehensive performance report
    
    Parameters:
    -----------
    models_results : dict
        Dictionary containing results for each model
    baseline_accuracy : float
        Baseline accuracy to compare against
    save_path : str, optional
        Path to save the report as HTML
    """
    report = []
    report.append("# Bank Marketing Classification - Model Performance Report\n")
    report.append(f"**Baseline Accuracy**: {baseline_accuracy:.4f}\n")
    report.append("\n## Model Comparison\n")
    
    # Create comparison table
    report.append("| Model | Train Time (s) | Train Accuracy | Test Accuracy | Improvement |")
    report.append("|-------|----------------|----------------|---------------|-------------|")
    
    for model_name, results in models_results.items():
        improvement = results['test_accuracy'] - baseline_accuracy
        report.append(f"| {model_name} | {results['train_time']:.3f} | "
                     f"{results['train_accuracy']:.4f} | {results['test_accuracy']:.4f} | "
                     f"{improvement:+.4f} |")
    
    report_text = "\n".join(report)
    
    if save_path:
        # Convert markdown to HTML if save_path is provided
        try:
            import markdown
            html = markdown.markdown(report_text, extensions=['tables'])
            with open(save_path, 'w') as f:
                f.write(f"<html><body>{html}</body></html>")
            print(f"Report saved to {save_path}")
        except ImportError:
            # Save as text if markdown package not available
            with open(save_path.replace('.html', '.md'), 'w') as f:
                f.write(report_text)
            print(f"Report saved as markdown to {save_path.replace('.html', '.md')}")
    
    return report_text

def plot_randomized_search_results(random_search, title="RandomizedSearchCV Results"):
    """Plot results from RandomizedSearchCV"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Extract results
    results_df = pd.DataFrame(random_search.cv_results_)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} - Parameter Exploration', fontsize=16)
    
    # 1. Score distribution
    ax1 = axes[0, 0]
    scores = results_df['mean_test_score']
    ax1.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(random_search.best_score_, color='red', linestyle='--', linewidth=2, label=f'Best: {random_search.best_score_:.4f}')
    ax1.set_xlabel('Cross-validation Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of CV Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter importance (top 10 configurations)
    ax2 = axes[0, 1]
    top_10_idx = results_df.nlargest(10, 'mean_test_score').index
    top_10_scores = results_df.loc[top_10_idx, 'mean_test_score']
    colors = ['red' if i == results_df['mean_test_score'].idxmax() else 'skyblue' for i in top_10_idx]
    
    bars = ax2.bar(range(len(top_10_scores)), top_10_scores, color=colors)
    ax2.set_xlabel('Configuration Rank')
    ax2.set_ylabel('CV Score')
    ax2.set_title('Top 10 Configurations')
    ax2.set_xticks(range(len(top_10_scores)))
    ax2.set_xticklabels([f'#{i+1}' for i in range(len(top_10_scores))])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, top_10_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Parameter correlation heatmap
    ax3 = axes[1, 0]
    # Get numeric parameters only
    param_cols = [col for col in results_df.columns if col.startswith('param_') and 
                  col not in ['param_criterion', 'param_max_features']]
    
    if param_cols:
        # Convert to numeric where possible
        numeric_params = pd.DataFrame()
        for col in param_cols:
            try:
                numeric_params[col.replace('param_', '')] = pd.to_numeric(results_df[col])
            except:
                pass
        
        if not numeric_params.empty:
            numeric_params['score'] = results_df['mean_test_score']
            corr = numeric_params.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, ax=ax3, fmt='.3f')
            ax3.set_title('Parameter Correlation with Score')
    
    # 4. Convergence plot
    ax4 = axes[1, 1]
    # Calculate running maximum
    running_max = np.maximum.accumulate(results_df['mean_test_score'])
    iterations = range(1, len(running_max) + 1)
    
    ax4.plot(iterations, running_max, 'b-', linewidth=2, label='Best score found')
    ax4.scatter(range(1, len(scores) + 1), scores, alpha=0.3, s=30, c='gray', label='Individual scores')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Best CV Score')
    ax4.set_title('Convergence Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print best parameters
    print("\nBest Parameters Found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest Cross-validation Score: {random_search.best_score_:.4f}")
    print(f"Number of iterations: {len(results_df)}")


def plot_grid_vs_random_search(grid_results, random_results, grid_time, random_time):
    """Compare GridSearchCV vs RandomizedSearchCV results"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Comparison of scores and times
    methods = ['GridSearchCV', 'RandomizedSearchCV']
    scores = [grid_results.best_score_, random_results.best_score_]
    times = [grid_time, random_time]
    n_configs = [len(grid_results.cv_results_['mean_test_score']), 
                 len(random_results.cv_results_['mean_test_score'])]
    
    # Score comparison
    bars1 = ax1.bar(methods, scores, color=['steelblue', 'darkorange'])
    ax1.set_ylabel('Best CV Score')
    ax1.set_title('Best Scores Comparison')
    ax1.set_ylim(min(scores) * 0.99, max(scores) * 1.01)
    
    # Add value labels
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{score:.4f}', ha='center', va='bottom')
    
    # Time and configurations comparison
    x = np.arange(len(methods))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, times, width, label='Time (seconds)', color='lightcoral')
    bars3 = ax2.bar(x + width/2, n_configs, width, label='Configurations tested', color='lightgreen')
    
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Value')
    ax2.set_title('Efficiency Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    
    # Add value labels
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=9)
    
    for bar, n_config in zip(bars3, n_configs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{n_config}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print efficiency metrics
    print("\nEfficiency Metrics:")
    print(f"GridSearchCV: {times[0]/n_configs[0]:.3f} seconds per configuration")
    print(f"RandomizedSearchCV: {times[1]/n_configs[1]:.3f} seconds per configuration")
    print(f"\nSpeedup: {times[0]/times[1]:.1f}x faster with RandomizedSearchCV")