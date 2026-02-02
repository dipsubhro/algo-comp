"""
Simple Genetic Algorithm - Optimized for Good Results
"""

import numpy as np
import random
import time
from multiprocessing import Pool, cpu_count
from benchmark_functions import benchmark_functions, function_bounds
from config import *

def initialize_population(bounds):
    """Initialize population - mix of random and near-zero"""
    min_v, max_v = bounds
    pop = []
    
    # Half random, half near optimal
    for i in range(POP_SIZE):
        if i < POP_SIZE // 2:
            pop.append(np.random.uniform(min_v, max_v, NUM_DIMENSIONS))
        else:
            pop.append(np.random.normal(0, 0.5, NUM_DIMENSIONS))
    
    return np.array(pop)

def tournament_selection(pop, fitness, k=3):
    """Select best from k random individuals"""
    indices = np.random.choice(len(pop), k, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx].copy()

def crossover(p1, p2):
    """Simple blend crossover"""
    alpha = np.random.rand(NUM_DIMENSIONS)
    return alpha * p1 + (1 - alpha) * p2

def mutate(x, bounds, generation):
    """Adaptive mutation - gets smaller over time"""
    min_v, max_v = bounds
    strength = 0.5 * (1 - generation / NUM_GENERATIONS)
    
    for i in range(NUM_DIMENSIONS):
        if np.random.rand() < MUTATION_RATE:
            x[i] += np.random.normal(0, strength)
    
    return np.clip(x, min_v, max_v)

def run_ga(func_name, seed=42, return_history=False):
    """Run genetic algorithm"""
    np.random.seed(seed)
    random.seed(seed)
    
    func = benchmark_functions[func_name]
    bounds = function_bounds[func_name]
    
    pop = initialize_population(bounds)
    best_value = np.inf
    history = [] if return_history else None
    
    for gen in range(NUM_GENERATIONS):
        # Evaluate
        fitness = np.array([func(ind) for ind in pop])
        best_value = min(best_value, fitness.min())
        
        if return_history:
            history.append(best_value)
        
        # New population
        new_pop = []
        
        # Keep best (elitism)
        elite_idx = np.argsort(fitness)[:ELITE_SIZE]
        for idx in elite_idx:
            new_pop.append(pop[idx].copy())
        
        # Create offspring
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            
            if np.random.rand() < CROSSOVER_RATE:
                child = crossover(p1, p2)
            else:
                child = p1.copy()
            
            child = mutate(child, bounds, gen)
            new_pop.append(child)
        
        pop = np.array(new_pop[:POP_SIZE])
    
    if return_history:
        return best_value, history
    return best_value

def run_multiple_experiments(func_name, num_runs=NUM_RUNS):
    """Run multiple times and collect results"""
    results = []
    history = None
    
    print(f"  Running {func_name}: ", end="", flush=True)
    
    for run in range(num_runs):
        seed = 12345 + run * 9876
        
        if run == 0:
            best, hist = run_ga(func_name, seed, return_history=True)
            history = hist
        else:
            best = run_ga(func_name, seed, return_history=False)
        
        results.append(best)
        
        # Show progress at 10, 20, 30
        if (run + 1) % 10 == 0:
            print(f"{run + 1}", end=" ", flush=True)
    
    print("✓")
    return results, history

def _run_single_function_worker(args):
    """Worker for parallel execution"""
    func_name, num_runs = args
    results, history = run_multiple_experiments(func_name, num_runs)
    return (func_name, results, history)

def run_all_functions_parallel(num_runs=NUM_RUNS):
    """Run all functions in parallel"""
    start_time = time.time()
    
    all_functions = list(benchmark_functions.keys())
    total_functions = len(all_functions)
    
    print("="*70)
    print("🚀 PARALLEL EXECUTION - ALL FUNCTIONS")
    print("="*70)
    print(f"Functions: {total_functions}")
    print(f"Runs per function: {num_runs}")
    print(f"Total runs: {total_functions * num_runs}")
    print(f"Population: {POP_SIZE}")
    print(f"Dimensions: {NUM_DIMENSIONS}")
    print(f"Generations: {NUM_GENERATIONS}")
    print(f"CPU cores: {cpu_count()}")
    print(f"Time limit: {MAX_EXECUTION_TIME}s")
    print("="*70)
    print()
    
    worker_args = [(func_name, num_runs) for func_name in all_functions]
    num_workers = min(cpu_count(), total_functions)
    
    all_results = {}
    plot_data = {}
    
    with Pool(processes=num_workers) as pool:
        results_list = pool.map(_run_single_function_worker, worker_args)
    
    for func_name, results, history in results_list:
        all_results[func_name] = results
        plot_data[func_name] = history
    
    execution_time = time.time() - start_time
    
    print()
    print("="*70)
    print(f"✓ ALL COMPLETED IN {execution_time:.2f}s!")
    
    if execution_time <= MAX_EXECUTION_TIME:
        print(f"✓ Target met: {execution_time:.2f}s ≤ {MAX_EXECUTION_TIME}s")
    else:
        print(f"⚠️  Time: {execution_time:.2f}s (target was {MAX_EXECUTION_TIME}s)")
    
    print(f"✓ Average per function: {execution_time/total_functions:.2f}s")
    print("="*70)
    print()
    
    return all_results, plot_data, execution_time

def calculate_statistics(results):
    """Calculate stats"""
    results_array = np.array(results)
    return {
        'min': np.min(results_array),
        'mean': np.mean(results_array),
        'median': np.median(results_array),
        'std': np.std(results_array),
        'max': np.max(results_array)
    }