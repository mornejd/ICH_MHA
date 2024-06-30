import random
import numpy as np
from Objective_function import evaluate_solution

def simulated_annealing(X_train, y_train, X_test, y_test, max_iterations=50, temp=100):
    # Define the parameter space for CatBoost
    learning_rate_options = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    depth_options = [4, 6, 8, 10, 12, 14, 16]
    l2_leaf_reg_options = [1, 3, 5, 7, 9, 12, 15]
    iterations_options = [50, 100, 150, 200, 300, 400, 500]

    # Initial solution
    current_solution = {
        'learning_rate': random.choice(learning_rate_options),
        'depth': random.choice(depth_options),
        'l2_leaf_reg': random.choice(l2_leaf_reg_options),
        'iterations': random.choice(iterations_options)
    }

    current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc = evaluate_solution(current_solution, X_train, y_train, X_test, y_test)
   
    best_solution = current_solution
    best_scores = (current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc)
    
    best_scores_over_time = []
    for i in range(max_iterations):
        # Create new solution
        new_solution = {
            'learning_rate': random.choice(learning_rate_options),
            'depth': random.choice(depth_options),
            'l2_leaf_reg': random.choice(l2_leaf_reg_options),
            'iterations': random.choice(iterations_options)
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
