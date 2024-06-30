import numpy as np
import random
from Objective_function import evaluate_solution

def initialize_population(size, num_layers_options, units_options, activation_options, optimizer_options, loss_options, batch_size_options, epochs_options):
    population = []
    for _ in range(size):
        num_layers = random.choice(num_layers_options)
        units_per_layer = [random.choice(units_options) for _ in range(num_layers)]
        activation_per_layer = [random.choice(activation_options) for _ in range(num_layers)]
        individual = {
            'num_layers': num_layers,
            'units_per_layer': units_per_layer,
            'activation_per_layer': activation_per_layer,
            'optimizer': random.choice(optimizer_options),
            'loss': random.choice(loss_options),
            'batch_size': random.choice(batch_size_options),
            'epochs': random.choice(epochs_options)
        }
        population.append(individual)
    return population

def calculate_fitness(solution, X_train, y_train, X_test, y_test):
    _, precision, recall, f1, sensitivity, specificity, pr_auc, roc_auc = evaluate_solution(solution, X_train, y_train, X_test, y_test)
    return f1  # Using F1 score as the fitness measure

def select_parents(population, fitnesses, num_parents):
    parents = list(np.argsort(fitnesses)[-num_parents:])
    return [population[i] for i in parents]

def crossover(parents, offspring_size):
    offspring = []
    for k in range(offspring_size):
        parent1_idx = k % len(parents)
        parent2_idx = (k + 1) % len(parents)
        parent1, parent2 = parents[parent1_idx], parents[parent2_idx]
        num_layers = random.choice([parent1['num_layers'], parent2['num_layers']])
        units_per_layer = [random.choice([parent1['units_per_layer'][i % len(parent1['units_per_layer'])], parent2['units_per_layer'][i % len(parent2['units_per_layer'])]]) for i in range(num_layers)]
        activation_per_layer = [random.choice([parent1['activation_per_layer'][i % len(parent1['activation_per_layer'])], parent2['activation_per_layer'][i % len(parent2['activation_per_layer'])]]) for i in range(num_layers)]
        child = {
            'num_layers': num_layers,
            'units_per_layer': units_per_layer,
            'activation_per_layer': activation_per_layer,
            'optimizer': random.choice([parent1['optimizer'], parent2['optimizer']]),
            'loss': random.choice([parent1['loss'], parent2['loss']]),
            'batch_size': random.choice([parent1['batch_size'], parent2['batch_size']]),
            'epochs': random.choice([parent1['epochs'], parent2['epochs']])
        }
        offspring.append(child)
    return offspring

def mutate(offspring_crossover, mutation_rate, num_layers_options, units_options, activation_options, optimizer_options, loss_options, batch_size_options, epochs_options):
    # Define a dictionary to map parameters to their options
    parameter_options = {
        'optimizer': optimizer_options,
        'loss': loss_options,
        'batch_size': batch_size_options,
        'epochs': epochs_options
    }

    for idx in range(len(offspring_crossover)):
        if random.random() < mutation_rate:
            # Choose a parameter to mutate
            parameter_to_mutate = random.choice(list(parameter_options.keys()) + ['num_layers', 'units_per_layer', 'activation_per_layer'])
            
            if parameter_to_mutate == 'num_layers':
                # If num_layers is mutated, units_per_layer and activation_per_layer must be adjusted
                new_num_layers = random.choice(num_layers_options)
                offspring_crossover[idx]['num_layers'] = new_num_layers
                offspring_crossover[idx]['units_per_layer'] = [random.choice(units_options) for _ in range(new_num_layers)]
                offspring_crossover[idx]['activation_per_layer'] = [random.choice(activation_options) for _ in range(new_num_layers)]
            
            elif parameter_to_mutate in ['units_per_layer', 'activation_per_layer']:
                # Only mutate these if num_layers stays constant
                current_num_layers = offspring_crossover[idx]['num_layers']
                if parameter_to_mutate == 'units_per_layer':
                    offspring_crossover[idx]['units_per_layer'] = [random.choice(units_options) for _ in range(current_num_layers)]
                else:
                    offspring_crossover[idx]['activation_per_layer'] = [random.choice(activation_options) for _ in range(current_num_layers)]
            
            else:
                # Mutate other parameters normally
                new_value = random.choice(parameter_options[parameter_to_mutate])
                offspring_crossover[idx][parameter_to_mutate] = new_value

    return offspring_crossover

def genetic_algorithm(X_train, y_train, X_test, y_test, num_generations, population_size, num_parents_mating, mutation_rate):
    num_layers_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    units_options = [8, 16, 32, 64, 128, 256, 512, 1024]
    activation_options = ['relu', 'sigmoid', 'tanh', 'elu', 'softmax', 'softplus', 'selu', 'softsign']
    optimizer_options = ['adam', 'sgd', 'rmsprop', 'adamax', 'nadam', 'adagrad', 'adadelta', 'ftrl']
    loss_options = ['binary_crossentropy', 'hinge', 'squared_hinge']
    batch_size_options = [8, 16, 32, 64, 128, 256, 512]
    epochs_options = [10, 30, 50, 75, 100, 150, 200, 300, 500]

    population = initialize_population(population_size, num_layers_options, units_options, activation_options, optimizer_options, loss_options, batch_size_options, epochs_options)
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
        mutated_offspring = mutate(offspring, mutation_rate, num_layers_options, units_options, activation_options, optimizer_options, loss_options, batch_size_options, epochs_options)
        population = parents + mutated_offspring

        print(f'Generation {generation + 1} | Best Fitness: {best_scores[0]}')

    return best_solution, best_scores, best_scores_over_time
