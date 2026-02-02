import numpy as np
import random
import time
from multiprocessing import Pool, cpu_count
BASE_SEED = 54321
DEFAULT_CONFIG = {'pop_size': 50, 'num_generations': 1000, 'crossover_rate': 0.9, 'mutation_rate': 0.15, 'elite_size': 3}

def initialize_population(bounds, n_dimensions, pop_size):
    min_v, max_v = bounds
    pop = []
    for i in range(pop_size):
        if i < pop_size // 2:
            pop.append(np.random.uniform(min_v, max_v, n_dimensions))
        else:
            pop.append(np.random.normal(0, 0.5, n_dimensions))
    return np.array(pop)

def tournament_selection(pop, fitness, k=3):
    indices = np.random.choice(len(pop), k, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx].copy()

def crossover(p1, p2, n_dimensions):
    alpha = np.random.rand(n_dimensions)
    return alpha * p1 + (1 - alpha) * p2

def mutate(x, bounds, generation, num_generations, mutation_rate):
    min_v, max_v = bounds
    strength = 0.5 * (1 - generation / num_generations)
    n_dimensions = len(x)
    for i in range(n_dimensions):
        if np.random.rand() < mutation_rate:
            x[i] += np.random.normal(0, strength)
    return np.clip(x, min_v, max_v)

def run_ga_single(func, bounds, n_dimensions, seed=42, return_history=False, pop_size=50, num_generations=1000, crossover_rate=0.9, mutation_rate=0.15, elite_size=3):
    np.random.seed(seed)
    random.seed(seed)
    pop = initialize_population(bounds, n_dimensions, pop_size)
    best_value = np.inf
    best_x = None
    history = [] if return_history else None
    for gen in range(num_generations):
        fitness = np.array([func(ind) for ind in pop])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_value:
            best_value = fitness[min_idx]
            best_x = pop[min_idx].copy()
        if return_history:
            history.append(best_value)
        new_pop = []
        elite_idx = np.argsort(fitness)[:elite_size]
        for idx in elite_idx:
            new_pop.append(pop[idx].copy())
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            if np.random.rand() < crossover_rate:
                child = crossover(p1, p2, n_dimensions)
            else:
                child = p1.copy()
            child = mutate(child, bounds, gen, num_generations, mutation_rate)
            new_pop.append(child)
        pop = np.array(new_pop[:pop_size])
    if return_history:
        return (best_value, best_x, history)
    return (best_value, best_x)

def run_ga(func, bounds, n_runs=30, n_dimensions=5, base_seed=None, pop_size=50, num_generations=1000, crossover_rate=0.9, mutation_rate=0.15, elite_size=3):
    if base_seed is None:
        base_seed = BASE_SEED
    results = []
    best_overall = np.inf
    best_x_overall = None
    convergence_history = None
    for run in range(n_runs):
        seed = base_seed + run * 9876
        if run == 0:
            best, best_x, history = run_ga_single(func, bounds, n_dimensions, seed, return_history=True, pop_size=pop_size, num_generations=num_generations, crossover_rate=crossover_rate, mutation_rate=mutation_rate, elite_size=elite_size)
            convergence_history = history
        else:
            best, best_x = run_ga_single(func, bounds, n_dimensions, seed, return_history=False, pop_size=pop_size, num_generations=num_generations, crossover_rate=crossover_rate, mutation_rate=mutation_rate, elite_size=elite_size)
        results.append(best)
        if best < best_overall:
            best_overall = best
            best_x_overall = best_x
    results_array = np.array(results)
    return {'best_f': np.min(results_array), 'avg_f': np.mean(results_array), 'median_f': np.median(results_array), 'max_f': np.max(results_array), 'std_f': np.std(results_array), 'best_x': best_x_overall, 'convergence_history': convergence_history, 'all_results': results}

def calculate_statistics(results):
    results_array = np.array(results)
    return {'min': np.min(results_array), 'mean': np.mean(results_array), 'median': np.median(results_array), 'std': np.std(results_array), 'max': np.max(results_array)}

def run_ga_legacy(func_name, seed=42, return_history=False):
    try:
        from benchmark_functions import benchmark_functions, function_bounds
        func = benchmark_functions[func_name]
        bounds = function_bounds[func_name]
        from config import NUM_DIMENSIONS, POP_SIZE, NUM_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE, ELITE_SIZE
        if return_history:
            best, _, history = run_ga_single(func, bounds, NUM_DIMENSIONS, seed, return_history=True, pop_size=POP_SIZE, num_generations=NUM_GENERATIONS, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE, elite_size=ELITE_SIZE)
            return (best, history)
        else:
            best, _ = run_ga_single(func, bounds, NUM_DIMENSIONS, seed, return_history=False, pop_size=POP_SIZE, num_generations=NUM_GENERATIONS, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE, elite_size=ELITE_SIZE)
            return best
    except ImportError:
        raise ImportError('Legacy mode requires benchmark_functions.py and config.py')

def run_multiple_experiments(func_name, num_runs=30):
    try:
        from benchmark_functions import benchmark_functions, function_bounds
        from config import NUM_DIMENSIONS, POP_SIZE, NUM_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE, ELITE_SIZE
        func = benchmark_functions[func_name]
        bounds = function_bounds[func_name]
        result = run_ga(func, bounds, n_runs=num_runs, n_dimensions=NUM_DIMENSIONS, pop_size=POP_SIZE, num_generations=NUM_GENERATIONS, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE, elite_size=ELITE_SIZE)
        return (result['all_results'], result['convergence_history'])
    except ImportError:
        raise ImportError('Legacy mode requires benchmark_functions.py and config.py')