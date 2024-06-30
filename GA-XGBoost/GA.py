import numpy as np
import random
from Objective_function import evaluate_solution

def initialize_population(size, C_options, dual_options, loss_options, penalty_options, tol_options, learning_rate_options, max_depth_options, max_features_options, min_samples_leaf_options, min_samples_split_options, n_estimators_options, subsample_options):
    return [{
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
    } for _ in range(size)]

def calculate_fitness(solution, X_train, y_train, X_test, y_test):
    return evaluate_solution(solution, X_train, y_train, X_test, y_test)[0]  # Return accuracy as fitness

def select_parents(population, fitnesses, num_parents):
    parents = list(np.argsort(fitnesses)[-num_parents:])
    return [population[i] for i in parents]

def crossover(parents, offspring_size):
    offspring = []
    crossover_point = np.random.randint(1, len(parents[0])-1)
    for k in range(offspring_size):
        parent1_idx = k % len(parents)
        parent2_idx = (k+1) % len(parents)
        offspring.append({**parents[parent1_idx], **{k: parents[parent2_idx][k] for k in list(parents[parent2_idx])[crossover_point:]}})
    return offspring

def mutate(offspring_crossover, mutation_rate, C_options, dual_options, loss_options, penalty_options, tol_options, learning_rate_options, max_depth_options, max_features_options, min_samples_leaf_options, min_samples_split_options, n_estimators_options, subsample_options):
    for idx in range(len(offspring_crossover)):
        if random.random() < mutation_rate:
            parameter_to_mutate = random.choice(['C', 'dual', 'loss', 'penalty', 'tol', 'learning_rate', 'max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split', 'n_estimators', 'subsample'])
            new_value = random.choice(eval(parameter_to_mutate + '_options'))
            offspring_crossover[idx][parameter_to_mutate] = new_value
    return offspring_crossover

def genetic_algorithm(X_train, y_train, X_test, y_test, num_generations, population_size, num_parents_mating, mutation_rate):
    C_options = [0.01, 0.1, 1, 5, 10, 20, 100]
    dual_options = [False, True]
    loss_options = ['log_loss']
    penalty_options = ['l2', 'none']
    tol_options = [1e-05, 1e-4, 1e-3, 1e-2]
    learning_rate_options = [0.01, 0.05, 0.1, 0.2]
    max_depth_options = [3, 4, 5, 6, 7, 8, 9, 10]
    max_features_options = [None, 'sqrt', 'log2']
    min_samples_leaf_options = [1, 2, 4]
    min_samples_split_options = [1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    n_estimators_options = [100, 200, 300, 400, 500]
    subsample_options = [0.4, 0.5, 0.7, 1.0]

    population = initialize_population(population_size, C_options, dual_options, loss_options, penalty_options, tol_options, learning_rate_options, max_depth_options, max_features_options, min_samples_leaf_options, min_samples_split_options, n_estimators_options, subsample_options)
    best_scores_over_time = []
    best_solution = None
    best_scores = None

    for generation in range(num_generations):
        fitnesses = [calculate_fitness(individual, X_train, y_train, X_test, y_test) for individual in population]
        best_fitness_idx = np.argmax(fitnesses)
        
        if not best_solution or fitnesses[best_fitness_idx] > (best_scores[0] if best_scores else 0):
            best_solution = population[best_fitness_idx]
            best_scores = evaluate_solution(best_solution, X_train, y_train, X_test, y_test)
        
        best_scores_over_time.append(best_scores[0])  # Assuming the first score in the tuple is the fitness score

        parents = select_parents(population, fitnesses, num_parents_mating)
        offspring_crossover = crossover(parents, offspring_size=population_size - num_parents_mating)
        offspring_mutation = mutate(offspring_crossover, mutation_rate, C_options, dual_options, loss_options, penalty_options, tol_options, learning_rate_options, max_depth_options, max_features_options, min_samples_leaf_options, min_samples_split_options, n_estimators_options, subsample_options)
        population = parents + offspring_mutation

        print(f'Generation {generation + 1} | Best Fitness: {best_scores[0]}')

    return best_solution, best_scores, best_scores_over_time

# Example of running the GA
# best_solution, best_fitness_over_generations = genetic_algorithm(X_train, y_train, X_test, y_test, 10, 20, 10, 0.1)
