import numpy as np
import random
from Objective_function import evaluate_solution

def initialize_population(size, solver_options, n_components_options, store_covariance_options, tol_options, shrinkage_options):
    return [{
        'solver': random.choice(solver_options),
        'n_components': random.choice(n_components_options),
        'store_covariance': random.choice(store_covariance_options),
        'tol': random.choice(tol_options),
        'shrinkage': random.choice(shrinkage_options)
    } for _ in range(size)]

def calculate_fitness(solution, X_train, y_train, X_test, y_test):
    return evaluate_solution(solution, X_train, y_train, X_test, y_test)[0]  # Return accuracy as fitness

def select_parents(population, fitnesses, num_parents):
    parents = list(np.argsort(fitnesses)[-num_parents:])
    return [population[i] for i in parents]

def crossover(parents, offspring_size):
    offspring = []
    for k in range(offspring_size):
        parent1_idx = k % len(parents)
        parent2_idx = (k + 1) % len(parents)
        child = {key: random.choice([parents[parent1_idx][key], parents[parent2_idx][key]]) for key in parents[0]}
        offspring.append(child)
    return offspring

def mutate(offspring_crossover, mutation_rate, solver_options, n_components_options, store_covariance_options, tol_options, shrinkage_options):
    for idx in range(len(offspring_crossover)):
        if random.random() < mutation_rate:
            parameter_to_mutate = random.choice(['solver', 'n_components', 'store_covariance', 'tol', 'shrinkage'])
            new_value = random.choice(eval(parameter_to_mutate + '_options'))
            offspring_crossover[idx][parameter_to_mutate] = new_value
    return offspring_crossover

def genetic_algorithm(X_train, y_train, X_test, y_test, num_generations, population_size, num_parents_mating, mutation_rate):
    solver_options = ['svd', 'lsqr', 'eigen']
    n_components_options = [0, 1]
    store_covariance_options = [True, False]
    tol_options = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
    shrinkage_options = [None, 'auto'] + [i / 10.0 for i in range(0, 11)]

    population = initialize_population(population_size, solver_options, n_components_options, store_covariance_options, tol_options, shrinkage_options)
    best_scores_over_time = []
    best_solution = None
    best_scores = None

    for generation in range(num_generations):
        fitnesses = [calculate_fitness(individual, X_train, y_train, X_test, y_test) for individual in population]
        best_fitness_idx = np.argmax(fitnesses)
        
        if not best_solution or fitnesses[best_fitness_idx] > (best_scores[0] if best_scores else 0):
            best_solution = population[best_fitness_idx]
            best_scores = evaluate_solution(best_solution, X_train, y_train, X_test, y_test)
        
        best_scores_over_time.append(best_scores[0])

        parents = select_parents(population, fitnesses, num_parents_mating)
        offspring = crossover(parents, offspring_size=population_size - num_parents_mating)
        mutated_offspring = mutate(offspring, mutation_rate, solver_options, n_components_options, store_covariance_options, tol_options, shrinkage_options)
        population = parents + mutated_offspring

        print(f'Generation {generation + 1} | Best Fitness: {best_scores[0]}')

    return best_solution, best_scores, best_scores_over_time
