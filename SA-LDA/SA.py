import random
import numpy as np
from Objective_function import evaluate_solution

def simulated_annealing(X_train, y_train, X_test, y_test, max_iterations=50, temp=100):
    # Define the parameter space for LDA
    solver_options = ['svd', 'lsqr', 'eigen']
    n_components_options = [0, 1]
    store_covariance_options = [True, False]
    tol_options = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
    shrinkage_options = [None, 'auto'] + [i/10.0 for i in range(0, 11)]

    # Initial solution
    current_solution = {
        'solver': random.choice(solver_options),
        'n_components': random.choice(n_components_options),
        'store_covariance': random.choice(store_covariance_options),
        'tol': random.choice(tol_options),
        'shrinkage': random.choice(shrinkage_options)
    }

    current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc = evaluate_solution(current_solution, X_train, y_train, X_test, y_test)
   
    best_solution = current_solution
    best_scores = (current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc)
    
    best_scores_over_time = []
    for i in range(max_iterations):
        # Create new solution
        new_solution = {
            'solver': random.choice(solver_options),
            'n_components': random.choice(n_components_options),
            'store_covariance': random.choice(store_covariance_options),
            'tol': random.choice(tol_options),
            'shrinkage': random.choice(shrinkage_options)
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
