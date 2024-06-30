import random
import numpy as np
from Objective_function import evaluate_solution

def simulated_annealing(X_train, y_train, X_test, y_test, max_iterations=50, temp=100):
    # Broader parameter space for binary classification
    num_layers_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    units_options = [8, 16, 32, 64, 128, 256, 512, 1024]
    activation_options = ['relu', 'sigmoid', 'tanh', 'elu', 'softmax', 'softplus', 'selu', 'softsign']
    optimizer_options = ['adam', 'sgd', 'rmsprop', 'adamax', 'nadam', 'adagrad', 'adadelta', 'ftrl']
    loss_options = ['binary_crossentropy', 'hinge', 'squared_hinge']
    batch_size_options = [8, 16, 32, 64, 128, 256, 512]
    epochs_options = [10, 30, 50, 75, 100, 150, 200, 300, 500]

    # Initial solution
    num_layers = random.choice(num_layers_options)
    units_per_layer = [random.choice(units_options) for _ in range(num_layers)]
    activation_per_layer = [random.choice(activation_options) for _ in range(num_layers)]
    current_solution = {
        'num_layers': num_layers,
        'units_per_layer': units_per_layer,
        'activation_per_layer': activation_per_layer,
        'optimizer': random.choice(optimizer_options),
        'loss': random.choice(loss_options),
        'batch_size': random.choice(batch_size_options),
        'epochs': random.choice(epochs_options)
    }

    current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc = evaluate_solution(current_solution, X_train, y_train, X_test, y_test)
   
    best_solution = current_solution
    best_scores = (current_score, current_precision, current_recall, current_f1, current_sensitivity, current_specificity, current_pr_auc, current_roc_auc)
    
    best_scores_over_time = []
    for i in range(max_iterations):
        # Create new solution        
        num_layers = random.choice(num_layers_options)
        units_per_layer = [random.choice(units_options) for _ in range(num_layers)]
        activation_per_layer = [random.choice(activation_options) for _ in range(num_layers)]
        new_solution = {
            'num_layers': num_layers,
            'units_per_layer': units_per_layer,
            'activation_per_layer': activation_per_layer,
            'optimizer': random.choice(optimizer_options),
            'loss': random.choice(loss_options),
            'batch_size': random.choice(batch_size_options),
            'epochs': random.choice(epochs_options)
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
