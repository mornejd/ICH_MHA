import random
import numpy as np
from Objective_function import evaluate_solution

def simulated_annealing(X_train, y_train, X_test, y_test, max_iterations=50, temp=100):
    # Define the parameter space for XGBoost
    C_options = [0.01, 0.1, 1, 10, 100]
    dual_options = [False, True]
    loss_options = ['log_loss']
    penalty_options = ['l2', 'none']
    tol_options = [1e-4, 1e-3, 1e-2]
    learning_rate_options = [0.01, 0.05, 0.1, 0.2]
    max_depth_options = [3, 4, 5, 6, 7, 8, 9, 10]
    max_features_options = [None, 'sqrt', 'log2']
    min_samples_leaf_options = [1, 2, 4]
    min_samples_split_options = [2, 5, 10]
    n_estimators_options = [100, 200, 300, 400, 500]
    subsample_options = [0.5, 0.7, 1.0]

    # Initial solution
    current_solution = {
        'C': random.choice(C_options),
        'dual': random.choice(dual_options),
        'loss': random.choice(loss_options),
        'penalty': random.choice(penalty_options),
        'tol': random.choice(tol_options),
        'learning_rate': random.choice(learning_rate_options),
        'max_depth': random.choice(max_depth_options),
        'max_features': random.choice(max_features_options),
        'min_samples_leaf': random.choice(min_samples_leaf_options),
        'min_samples_split': random.choice(min_samples_split_options),
        'n_estimators': random.choice(n_estimators_options),
        'subsample': random.choice(subsample_options)
    }

    current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc = evaluate_solution(current_solution, X_train, y_train, X_test, y_test)
   
    best_solution = current_solution
    best_scores = (current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc)
    
    best_scores_over_time = []
    for i in range(max_iterations):
        # Create new solution
        new_solution = {
            'C': random.choice(C_options),
            'dual': random.choice(dual_options),
            'loss': random.choice(loss_options),
            'penalty': random.choice(penalty_options),
            'tol': random.choice(tol_options),
            'learning_rate': random.choice(learning_rate_options),
            'max_depth': random.choice(max_depth_options),
            'max_features': random.choice(max_features_options),
            'min_samples_leaf': random.choice(min_samples_leaf_options),
            'min_samples_split': random.choice(min_samples_split_options),
            'n_estimators': random.choice(n_estimators_options),
            'subsample': random.choice(subsample_options)
        }
        
        new_score, new_precision, new_recall, new_f1, new_sensitivity, new_specificity, new_pr_auc, new_roc_auc = evaluate_solution(new_solution, X_train, y_train, X_test, y_test)
        # Acceptance probability
        if new_score > current_score or np.exp((new_score - current_score) / temp) > random.random():
            current_solution, current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc = new_solution, new_score, new_precision, new_recall, new_f1, new_sensitivity, new_specificity, new_pr_auc, new_roc_auc
            if current_score > best_scores[0]:
                best_solution = current_solution
                best_scores = (current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc)
        # Decrease temperature
        temp *= 0.95
        print(f"Iteration {i + 1}: Best Accuracy = {best_scores[0]}, Precision = {best_scores[1]}, Recall = {best_scores[2]}, F1 Score = {best_scores[3]}, Sensitivity = {best_scores[4]}, Specificity = {best_scores[5]}, pr_auc = {best_scores[6]}, roc_auc = {best_scores[7]}")
        best_scores_over_time.append(best_scores[0])
        
    return best_solution, best_scores, best_scores_over_time
