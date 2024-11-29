import random
import matplotlib.pyplot as plt


# Add this function to plot the best fitness values over generations
def plot_fitness_over_generations(best_fitness_values):
    plt.plot(range(1, len(best_fitness_values) + 1), best_fitness_values, marker='o')
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.xticks(range(1, len(best_fitness_values) + 1))  # Set x-axis ticks to integer values
    plt.show()

# Step 1: Initialize population
def initialize_population(size, num_bits):
    return [''.join(random.choice('01') for _ in range(num_bits)) for _ in range(size)]

# Step 2: Calculate fitness
def calculate_fitness(population):
    fitness = [int(individual, 2) ** 2 for individual in population]
    return fitness

# Step 3: Roulette Wheel Selection
def roulette_wheel_selection(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    selected = random.choices(population, weights=probabilities, k=2)
    return selected

# Step 4: Single-Point Crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Step 5: Mutation
def mutate(individual, mutation_rate):
    mutated = ''.join(
        bit if random.random() > mutation_rate else str(1 - int(bit))
        for bit in individual
    )
    return mutated

# Step 6: Genetic Algorithm
def genetic_algorithm(pop_size, num_bits, mutation_rate, max_no_improvement):
    population = initialize_population(pop_size, num_bits)
    generation = 0
    best_fitness_overall = -float("inf")
    no_improvement_count = 0
    best_fitness_values = []

    while True:
        generation += 1
        fitness = calculate_fitness(population)
        next_population = []
        for _ in range(pop_size // 2):
            # Selection
            parent1, parent2 = roulette_wheel_selection(population, fitness)
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_population.extend([child1, child2])

        # Update population
        population = next_population
        best_individual = max(population, key=lambda ind: int(ind, 2) ** 2)
        best_fitness_value = int(best_individual, 2) ** 2
        best_fitness_values.append(best_fitness_value)
        print(f"Generation {generation}: Best = {best_individual}, Fitness = {best_fitness_value}")

        # Check for improvement
        if best_fitness_value > best_fitness_overall:
            best_fitness_overall = best_fitness_value
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Stopping condition: No improvement
        if no_improvement_count >= max_no_improvement:
            print(f"No improvement for {max_no_improvement} generations. Stopping.")
            break

    # Plot the best fitness values over generations
    plot_fitness_over_generations(best_fitness_values)
    return best_individual, best_fitness_value, generation

# Run the Genetic Algorithm
population_size = 4
bit_str_len = 5
rate_of_mutation = 0.1
maximum_no_improvement = 10  # Stop if no improvement in 10 generations

best_solution, best_fitness, final_generation = genetic_algorithm(population_size, bit_str_len, rate_of_mutation, maximum_no_improvement)
print(f"Solution: {best_solution} (Decimal: {int(best_solution, 2)}), Fitness: {best_fitness}, Reached in Generation: {final_generation}")