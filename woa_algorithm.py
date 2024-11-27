import numpy as np

def initialize_population(size, dim, bounds):
    """
    Initialize a random population within the given bounds.
    :param size: Number of individuals in the population
    :param dim: Dimension of the solution
    :param bounds: Tuple of (lower_bound, upper_bound)
    :return: Population matrix
    """
    population = np.random.uniform(bounds[0], bounds[1], (size, dim))
    
    # Avoid population initialized exactly at the lower bound (60)
    population[population == bounds[0]] += np.random.uniform(10, 100)  # Slight adjustment to avoid exact lower bound

    return population

def fitness_function(solution, system_data):
    """
    Fitness function to calculate power losses (with penalty for solutions close to the lower bound).
    :param solution: Candidate DG sizes
    :param system_data: Additional parameters about the system
    :return: Calculated fitness value (lower is better)
    """
    resistances = system_data['resistances']
    num_buses = len(resistances)

    # Calculate the total loss (example: sum of DG sizes weighted by resistances)
    power_loss = np.sum(solution * resistances)  # Simplified power loss calculation

    # A mild penalty for DG sizes too close to 60, but not too strong
    penalty = np.sum((solution - 60) ** 2) / 1000  # Reducing penalty weight

    return power_loss + penalty

def update_position(current_position, best_position, a, A, C, rand_factor, bounds):
    """
    Update whale's position based on encircling prey or spiral movement.
    """
    if rand_factor < 0.5:
        new_position = best_position - A * np.abs(C * best_position - current_position)
    else:
        distance = np.abs(best_position - current_position)
        b = 1  # Archimedean spiral constant
        l = np.random.uniform(-1, 1)
        new_position = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + best_position
    
    # Ensure new position stays within the bounds
    return np.clip(new_position, bounds[0], bounds[1])

def WOA(system_data, population_size=50, iterations=100, bounds=(60, 3000)):
    """
    Whale Optimization Algorithm to minimize power losses.
    :param system_data: Input data for the system
    :param population_size: Number of whales in the population
    :param iterations: Maximum iterations
    :param bounds: Bounds for DG sizes
    :return: Best solution and its fitness
    """
    dim = len(system_data['resistances'])
    population = initialize_population(population_size, dim, bounds)
    fitness = np.array([fitness_function(ind, system_data) for ind in population])
    best_solution = population[np.argmin(fitness)]
    
    best_fitness = fitness.min()  # Keep track of the best fitness
    for iteration in range(iterations):
        # Slow down the decay of 'a' to allow better exploration
        a = 2 - (2 * iteration / iterations)  # Gradually decay from 2 to 0
        
        for i in range(population_size):
            A = 2 * a * np.random.random() - a
            C = 2 * np.random.random()
            rand_factor = np.random.random()
            population[i] = update_position(population[i], best_solution, a, A, C, rand_factor, bounds)
        
        # Recalculate fitness after position updates
        fitness = np.array([fitness_function(ind, system_data) for ind in population])
        current_best_solution = population[np.argmin(fitness)]
        current_best_fitness = fitness.min()

        # Debugging: print the best solution and fitness at each iteration
        print(f"Iteration {iteration}: Best Solution = {current_best_solution}, Fitness = {current_best_fitness}")

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution

    return best_solution, best_fitness
