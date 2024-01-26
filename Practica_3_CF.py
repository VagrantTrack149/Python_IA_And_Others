import random
import math

# Function to decode a binary string to a decimal value within the given interval
def decode(binary_string, interval):
    min_val, max_val = interval
    decimal_value = int(binary_string, 2) * (max_val - min_val) / (2**len(binary_string) - 1) + min_val
    return decimal_value

# Function to calculate the fitness of an individual
def fitness_function(x):
    return -(1 - x**2)  # Return the negative of the function value

# Function to generate an initial population of binary strings
def generate_population(size, chromosome_length):
    population = []
    for _ in range(size):
        # Generate a random binary string
        individual = ''.join(random.choice('01') for _ in range(chromosome_length))
        population.append(individual)
    return population

# Function to perform roulette wheel selection
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_index = random.choices(range(len(population)), probabilities)[0]
    return population[selected_index]

# Function to perform crossover
def crossover(parent1, parent2, crossover_prob):
    if random.random() < crossover_prob:
        # Generate a random crossover point
        crossover_point = random.randint(0, min(len(parent1), len(parent2)) - 1)
        # Perform crossover
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

# Function to evolve the population
def evolve(population, crossover_prob, mutation_chance, interval):
    new_population = []
    fitness_values = [fitness_function(decode(individual, interval)) for individual in population]

    # Generate offspring until the new population is of the same size as the current population
    while len(new_population) < len(population):
        # Select two parents using roulette wheel selection
        parent1 = roulette_wheel_selection(population, fitness_values)
        parent2 = roulette_wheel_selection(population, fitness_values)

        # Perform crossover to produce two children
        child1, child2 = crossover(parent1, parent2, crossover_prob)

        # Add the children to the new population
        new_population.append(child1)
        new_population.append(child2)

    return new_population

# Main genetic algorithm function
def genetic_algorithm(population_size, num_generations, chromosome_length, interval):
    population = generate_population(population_size, chromosome_length)

    for _ in range(num_generations):
        # Evolve the population
        population = evolve(population, crossover_prob=0.5, mutation_chance=0, interval=interval)

    # Find the individual with the highest fitness value in the final population
    decoded_population = [decode(individual, interval) for individual in population]
    fitness_values = [fitness_function(x) for x in decoded_population]
    best_individual = population[fitness_values.index(max(fitness_values))]
    best_fitness = fitness_function(decode(best_individual, interval))

    return best_individual, best_fitness

# Example usage
population_size = 1000
num_generations = 500
chromosome_length = 8
interval = (-5, 5)

best_individual, best_fitness = genetic_algorithm(population_size, num_generations, chromosome_length, interval)

decoded_best_individual = decode(best_individual, interval)

print("Binary maximum:", best_individual)
print("Decoded maximum:", decoded_best_individual)
print("Fitness evaluation at maximum:", fitness_function(decoded_best_individual))
print("Number of generations:", num_generations)