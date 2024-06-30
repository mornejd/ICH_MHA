import numpy as np
import random
from Objective_function import evaluate_solution

def initialize_population(size, learning_rate_options, depth_options, l2_leaf_reg_options, iterations_options):
    return [{
        'learning_rate': random.choice(learning_rate_options),
        'depth': random.choice(depth_options),
        'l2_leaf_reg': random.choice(l2_leaf_reg_options),
        'iterations': random.choice(iterations_options)
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

def mutate(offspring_crossover, mutation_rate, learning_rate_options, depth_options, l2_leaf_reg_options, iterations_options):
    for idx in range(len(offspring_crossover)):
        if random.random() < mutation_rate:
            parameter_to_mutate = random.choice(['learning_rate', 'depth', 'l2_leaf_reg', 'iterations'])
            new_value = random.choice(eval(parameter_to_mutate + '_options'))
            offspring_crossover[idx][parameter_to_mutate] = new_value
    return offspring_crossover

def genetic_algorithm(X_train, y_train, X_test, y_test, num_generations, population_size, num_parents_mating, mutation_rate):
    learning_rate_options = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    depth_options = [4, 6, 8, 10, 12, 14, 16]
    l2_leaf_reg_options = [1, 3, 5, 7, 9, 12, 15]
    iterations_options = [50, 100, 150, 200, 300, 400, 500]

    population = initialize_population(population_size, learning_rate_options, depth_options, l2_leaf_reg_options, iterations_options)
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
        offspring_mutation = mutate(offspring_crossover, mutation_rate, learning_rate_options, depth_options, l2_leaf_reg_options, iterations_options)
        population = parents + offspring_mutation

        print(f'Generation {generation + 1} | Best Fitness: {best_scores[0]}')

    return best_solution, best_scores, best_scores_over_time




# Example of running the GA
# best_solution, best_fitness_over_generations = genetic_algorithm(X_train, y_train, X_test, y_test, 10, 20, 10, 0.1)
