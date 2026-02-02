"""
Unified Algorithm Comparison Runner

Runs PSO, SA, GA, and Tabu Search on all benchmark functions and generates:
1. A unified comparison table
2. Convergence curves for each algorithm
3. A summary comparing all algorithms

Usage: uv run runner.py
"""

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add algorithm directories to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PSO"))
sys.path.insert(0, str(ROOT / "SA"))
sys.path.insert(0, str(ROOT / "TA"))
sys.path.insert(0, str(ROOT / "GA"))

# Import benchmark functions from global func.py
from func import (
    sphere, sum_of_squares, sum_of_different_powers, step, brown,
    zakharov, dixon_price, schumer_steiglitz, csendes, sixth_power,
    powell, quartic, rotated_hyper_ellipsoid, discus, exponential
)

# Import algorithms
from pso_algorithm import run_pso
from sa_algorithm import run_sa
from run_tabu import run_tabu

# Try to import GA
try:
    from ga_algorithm import run_ga
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False
    print("Warning: GA module not available for unified comparison")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_RUNS = 30
NUM_DIMENSIONS = 5
# Note: Each algorithm has its own BASE_SEED in its respective folder

# Algorithm-specific configurations (defaults)
PSO_CONFIG = {
    'n_particles': 40,
    'max_iterations': 1000,
}

SA_CONFIG = {
    'max_iterations': 1500,
    'initial_temp': 1000.0,
    'step_size': 0.5,
}

# Default Tabu config (used as fallback)
TABU_CONFIG = {
    'neighbors': 40,
    'tenure': 8,
    'max_iter': 2000,
}

GA_CONFIG = {
    'pop_size': 50,
    'num_generations': 1000,
    'crossover_rate': 0.9,
    'mutation_rate': 0.15,
    'elite_size': 3,
}

# GA configurations per function
GA_CONFIGS = {
    "Sphere": {'pop_size': 50, 'num_generations': 1000, 'crossover_rate': 0.9, 'mutation_rate': 0.15, 'elite_size': 3},
    "Sum_of_Squares": {'pop_size': 50, 'num_generations': 1000, 'crossover_rate': 0.9, 'mutation_rate': 0.15, 'elite_size': 3},
    "Sum_of_Diff_Powers": {'pop_size': 40, 'num_generations': 800, 'crossover_rate': 0.9, 'mutation_rate': 0.1, 'elite_size': 3},
    "Step": {'pop_size': 60, 'num_generations': 1200, 'crossover_rate': 0.9, 'mutation_rate': 0.2, 'elite_size': 5},
    "Brown": {'pop_size': 50, 'num_generations': 1000, 'crossover_rate': 0.9, 'mutation_rate': 0.15, 'elite_size': 3},
    "Zakharov": {'pop_size': 50, 'num_generations': 1000, 'crossover_rate': 0.9, 'mutation_rate': 0.15, 'elite_size': 3},
    "Dixon_Price": {'pop_size': 55, 'num_generations': 1200, 'crossover_rate': 0.9, 'mutation_rate': 0.15, 'elite_size': 4},
    "Schumer_Steiglitz": {'pop_size': 50, 'num_generations': 1000, 'crossover_rate': 0.9, 'mutation_rate': 0.15, 'elite_size': 3},
    "Csendes": {'pop_size': 40, 'num_generations': 800, 'crossover_rate': 0.9, 'mutation_rate': 0.1, 'elite_size': 3},
    "Sixth_Power": {'pop_size': 40, 'num_generations': 800, 'crossover_rate': 0.9, 'mutation_rate': 0.1, 'elite_size': 3},
    "Powell": {'pop_size': 60, 'num_generations': 1500, 'crossover_rate': 0.9, 'mutation_rate': 0.2, 'elite_size': 5},
    "Quartic": {'pop_size': 40, 'num_generations': 800, 'crossover_rate': 0.9, 'mutation_rate': 0.1, 'elite_size': 3},
    "Rotated_Hyper_Ellipsoid": {'pop_size': 60, 'num_generations': 1500, 'crossover_rate': 0.9, 'mutation_rate': 0.2, 'elite_size': 5},
    "Discus": {'pop_size': 60, 'num_generations': 1500, 'crossover_rate': 0.9, 'mutation_rate': 0.2, 'elite_size': 5},
    "Exponential": {'pop_size': 40, 'num_generations': 800, 'crossover_rate': 0.9, 'mutation_rate': 0.1, 'elite_size': 3},
}

# ==============================================================================
# FUNCTION-SPECIFIC ALGORITHM CONFIGURATIONS
# ==============================================================================

# PSO configurations per function (n_particles, max_iterations)
PSO_CONFIGS = {
    "Sphere": {'n_particles': 40, 'max_iterations': 1000},
    "Sum_of_Squares": {'n_particles': 40, 'max_iterations': 1000},
    "Sum_of_Diff_Powers": {'n_particles': 30, 'max_iterations': 800},
    "Step": {'n_particles': 50, 'max_iterations': 1200},
    "Brown": {'n_particles': 35, 'max_iterations': 1000},
    "Zakharov": {'n_particles': 40, 'max_iterations': 1000},
    "Dixon_Price": {'n_particles': 45, 'max_iterations': 1200},
    "Schumer_Steiglitz": {'n_particles': 40, 'max_iterations': 1000},
    "Csendes": {'n_particles': 30, 'max_iterations': 800},
    "Sixth_Power": {'n_particles': 30, 'max_iterations': 800},
    "Powell": {'n_particles': 50, 'max_iterations': 1500},
    "Quartic": {'n_particles': 30, 'max_iterations': 800},
    "Rotated_Hyper_Ellipsoid": {'n_particles': 50, 'max_iterations': 1500},
    "Discus": {'n_particles': 50, 'max_iterations': 1500},
    "Exponential": {'n_particles': 30, 'max_iterations': 800},
}

# SA configurations per function (max_iterations, initial_temp, step_size)
SA_CONFIGS = {
    "Sphere": {'max_iterations': 1500, 'initial_temp': 1000.0, 'step_size': 0.5},
    "Sum_of_Squares": {'max_iterations': 1500, 'initial_temp': 1000.0, 'step_size': 1.0},
    "Sum_of_Diff_Powers": {'max_iterations': 1200, 'initial_temp': 500.0, 'step_size': 0.1},
    "Step": {'max_iterations': 2000, 'initial_temp': 1500.0, 'step_size': 5.0},
    "Brown": {'max_iterations': 1500, 'initial_temp': 800.0, 'step_size': 0.3},
    "Zakharov": {'max_iterations': 1500, 'initial_temp': 1000.0, 'step_size': 0.8},
    "Dixon_Price": {'max_iterations': 1500, 'initial_temp': 1000.0, 'step_size': 1.0},
    "Schumer_Steiglitz": {'max_iterations': 1500, 'initial_temp': 1000.0, 'step_size': 1.0},
    "Csendes": {'max_iterations': 1200, 'initial_temp': 500.0, 'step_size': 0.1},
    "Sixth_Power": {'max_iterations': 1200, 'initial_temp': 500.0, 'step_size': 0.1},
    "Powell": {'max_iterations': 2000, 'initial_temp': 1200.0, 'step_size': 0.5},
    "Quartic": {'max_iterations': 1200, 'initial_temp': 500.0, 'step_size': 0.1},
    "Rotated_Hyper_Ellipsoid": {'max_iterations': 2000, 'initial_temp': 1500.0, 'step_size': 3.0},
    "Discus": {'max_iterations': 2000, 'initial_temp': 1500.0, 'step_size': 5.0},
    "Exponential": {'max_iterations': 1200, 'initial_temp': 500.0, 'step_size': 0.1},
}

# Tabu configurations per function (neighbors, tenure, max_iter)
TABU_CONFIGS = {
    "Sphere": {'neighbors': 40, 'tenure': 8, 'max_iter': 2000},
    "Sum_of_Squares": {'neighbors': 40, 'tenure': 8, 'max_iter': 2000},
    "Sum_of_Diff_Powers": {'neighbors': 30, 'tenure': 5, 'max_iter': 1500},
    "Step": {'neighbors': 50, 'tenure': 10, 'max_iter': 2500},
    "Brown": {'neighbors': 35, 'tenure': 7, 'max_iter': 2000},
    "Zakharov": {'neighbors': 40, 'tenure': 8, 'max_iter': 2000},
    "Dixon_Price": {'neighbors': 45, 'tenure': 9, 'max_iter': 2000},
    "Schumer_Steiglitz": {'neighbors': 40, 'tenure': 8, 'max_iter': 2000},
    "Csendes": {'neighbors': 30, 'tenure': 5, 'max_iter': 1500},
    "Sixth_Power": {'neighbors': 30, 'tenure': 5, 'max_iter': 1500},
    "Powell": {'neighbors': 50, 'tenure': 10, 'max_iter': 2500},
    "Quartic": {'neighbors': 30, 'tenure': 5, 'max_iter': 1500},
    "Rotated_Hyper_Ellipsoid": {'neighbors': 50, 'tenure': 10, 'max_iter': 2500},
    "Discus": {'neighbors': 50, 'tenure': 10, 'max_iter': 2500},
    "Exponential": {'neighbors': 30, 'tenure': 5, 'max_iter': 1500},
}

# Benchmark functions with their bounds
BENCHMARK_FUNCTIONS = [
    ("Sphere", sphere, (-5.12, 5.12), NUM_DIMENSIONS),
    ("Sum_of_Squares", sum_of_squares, (-10, 10), NUM_DIMENSIONS),
    ("Sum_of_Diff_Powers", sum_of_different_powers, (-1, 1), NUM_DIMENSIONS),
    ("Step", step, (-100, 100), NUM_DIMENSIONS),
    ("Brown", brown, (-1, 4), NUM_DIMENSIONS),
    ("Zakharov", zakharov, (-5, 10), NUM_DIMENSIONS),
    ("Dixon_Price", dixon_price, (-10, 10), NUM_DIMENSIONS),
    ("Schumer_Steiglitz", schumer_steiglitz, (-10, 10), NUM_DIMENSIONS),
    ("Csendes", csendes, (-1, 1), NUM_DIMENSIONS),
    ("Sixth_Power", sixth_power, (-1, 1), NUM_DIMENSIONS),
    ("Powell", powell, (-4, 5), 4),  # Works best with dims divisible by 4
    ("Quartic", quartic, (-1.28, 1.28), NUM_DIMENSIONS),
    ("Rotated_Hyper_Ellipsoid", rotated_hyper_ellipsoid, (-65.536, 65.536), NUM_DIMENSIONS),
    ("Discus", discus, (-100, 100), NUM_DIMENSIONS),
    ("Exponential", exponential, (-1, 1), NUM_DIMENSIONS),
]

# ==============================================================================
# ALGORITHM RUNNERS
# ==============================================================================

def run_pso_experiment(args):
    """Run PSO on a single function."""
    name, fn, bounds, dims = args
    start = time.time()
    logger.info(f"[PSO] Starting {name}...")
    
    # Get function-specific config or fall back to default
    config = PSO_CONFIGS.get(name, PSO_CONFIG)
    
    result = run_pso(
        fn, bounds,
        n_runs=NUM_RUNS,
        n_particles=config['n_particles'],
        n_dimensions=dims,
        max_iterations=config['max_iterations']
        # Uses PSO's own BASE_SEED from pso_algorithm.py
    )
    elapsed = time.time() - start
    logger.info(f"[PSO] {name} completed in {elapsed:.2f}s | Best: {result['best_f']:.4e}")
    return ("PSO", name, result)


def run_sa_experiment(args):
    """Run SA on a single function."""
    name, fn, bounds, dims = args
    start = time.time()
    logger.info(f"[SA] Starting {name}...")
    
    # Get function-specific config or fall back to default
    config = SA_CONFIGS.get(name, SA_CONFIG)
    
    result = run_sa(
        fn, bounds,
        n_runs=NUM_RUNS,
        n_dimensions=dims,
        max_iterations=config['max_iterations'],
        initial_temp=config['initial_temp'],
        step_size=config['step_size']
        # Uses SA's own BASE_SEED from sa_algorithm.py
    )
    elapsed = time.time() - start
    logger.info(f"[SA] {name} completed in {elapsed:.2f}s | Best: {result['best_f']:.4e}")
    return ("SA", name, result)


def run_tabu_experiment(args):
    """Run Tabu Search on a single function."""
    name, fn, bounds, dims = args
    start = time.time()
    logger.info(f"[Tabu] Starting {name}...")
    
    # Get function-specific config or fall back to default
    config = TABU_CONFIGS.get(name, TABU_CONFIG)
    
    result = run_tabu(
        fn,
        num_runs=NUM_RUNS,
        neighbors=config['neighbors'],
        tenure=config['tenure'],
        max_iter=config['max_iter'],
        bounds=bounds,
        dims=dims
    )
    elapsed = time.time() - start
    logger.info(f"[Tabu] {name} completed in {elapsed:.2f}s | Best: {result['best_f']:.4e}")
    return ("Tabu", name, result)


def run_ga_experiment(args):
    """Run GA on a single function."""
    if not GA_AVAILABLE:
        return None
    name, fn, bounds, dims = args
    start = time.time()
    logger.info(f"[GA] Starting {name}...")
    
    # Get function-specific config or fall back to default
    config = GA_CONFIGS.get(name, GA_CONFIG)
    
    try:
        result = run_ga(
            fn, bounds,
            n_runs=NUM_RUNS,
            n_dimensions=dims,
            # Uses GA's own BASE_SEED from ga_algorithm.py
            pop_size=config['pop_size'],
            num_generations=config['num_generations'],
            crossover_rate=config['crossover_rate'],
            mutation_rate=config['mutation_rate'],
            elite_size=config['elite_size']
        )
        elapsed = time.time() - start
        logger.info(f"[GA] {name} completed in {elapsed:.2f}s | Best: {result['best_f']:.4e}")
        return ("GA", name, result)
    except Exception as e:
        logger.error(f"[GA] {name} failed: {e}")
        return None


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_all_algorithms():
    """Run all algorithms on all benchmark functions.
    
    Algorithm order: PSO, GA, Tabu, SA
    """
    max_workers = max(1, os.cpu_count() - 2)
    total_start = time.time()
    
    logger.info("=" * 60)
    logger.info("UNIFIED ALGORITHM COMPARISON")
    logger.info("=" * 60)
    logger.info(f"Algorithms: PSO" + (", GA" if GA_AVAILABLE else "") + ", Tabu Search, SA")
    logger.info(f"Functions: {len(BENCHMARK_FUNCTIONS)}")
    logger.info(f"Runs per function: {NUM_RUNS}")
    logger.info(f"Dimensions: {NUM_DIMENSIONS}")
    logger.info(f"Workers: {max_workers}")
    logger.info("=" * 60)
    
    all_results = {}  # {algo: {func_name: result}}
    
    # Run PSO
    logger.info("")
    logger.info("▶ Starting PSO experiments...")
    pso_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pso_results = list(executor.map(run_pso_experiment, BENCHMARK_FUNCTIONS))
    all_results["PSO"] = {name: result for algo, name, result in pso_results}
    logger.info(f"✓ PSO completed in {time.time() - pso_start:.2f}s")
    
    # Run GA (before Tabu)
    if GA_AVAILABLE:
        logger.info("")
        logger.info("▶ Starting GA experiments...")
        ga_start = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            ga_results_raw = list(executor.map(run_ga_experiment, BENCHMARK_FUNCTIONS))
        ga_results_filtered = [r for r in ga_results_raw if r is not None]
        if ga_results_filtered:
            all_results["GA"] = {name: result for algo, name, result in ga_results_filtered}
        logger.info(f"✓ GA completed in {time.time() - ga_start:.2f}s")
    
    # Run Tabu Search
    logger.info("")
    logger.info("▶ Starting Tabu Search experiments...")
    tabu_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tabu_results = list(executor.map(run_tabu_experiment, BENCHMARK_FUNCTIONS))
    all_results["Tabu"] = {name: result for algo, name, result in tabu_results}
    logger.info(f"✓ Tabu Search completed in {time.time() - tabu_start:.2f}s")
    
    # Run SA (last)
    logger.info("")
    logger.info("▶ Starting SA experiments...")
    sa_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sa_results = list(executor.map(run_sa_experiment, BENCHMARK_FUNCTIONS))
    all_results["SA"] = {name: result for algo, name, result in sa_results}
    logger.info(f"✓ SA completed in {time.time() - sa_start:.2f}s")
    
    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info(f"🏁 All algorithms completed in {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
    
    return all_results


def generate_comparison_table(all_results):
    """Generate a unified comparison table matching the preview format.
    
    Format: Function | Algorithm | Best | Average | Std Dev | Rank
    Algorithm order: PSO, GA, Tabu, SA
    """
    # Define the algorithm order as specified
    algorithm_order = ["PSO", "GA", "Tabu", "SA"]
    # Filter to only include algorithms that are in the results
    algorithms = [algo for algo in algorithm_order if algo in all_results]
    
    func_names = [f[0] for f in BENCHMARK_FUNCTIONS]
    
    table_rows = []
    headers = ["Function", "Algorithm", "Best", "Average", "Std Dev", "Rank"]
    
    for func_name in func_names:
        # Calculate ranks for this function based on best values
        algo_best_values = []
        for algo in algorithms:
            if func_name in all_results[algo]:
                best_f = all_results[algo][func_name]['best_f']
                algo_best_values.append((algo, best_f))
        
        # Sort by best value to determine ranks
        algo_best_values.sort(key=lambda x: x[1])
        rank_map = {algo: rank + 1 for rank, (algo, _) in enumerate(algo_best_values)}
        
        # Add rows for each algorithm
        first_algo = True
        for algo in algorithms:
            if func_name in all_results[algo]:
                result = all_results[algo][func_name]
                best_f = result['best_f']
                avg_f = result['avg_f']
                std_f = result['std_f']
                rank = rank_map.get(algo, "N/A")
                
                # Only show function name for the first algorithm row
                func_display = func_name if first_algo else ""
                
                table_rows.append([
                    func_display,
                    algo,
                    f"{best_f:.2e}",
                    f"{avg_f:.2e}",
                    f"{std_f:.2e}",
                    rank
                ])
                first_algo = False
    
    return tabulate(table_rows, headers=headers, tablefmt="grid")


def generate_detailed_tables(all_results):
    """Generate detailed tables for each algorithm."""
    tables = {}
    
    for algo, results in all_results.items():
        table_rows = []
        headers = ["Rank", "Function", "Best f", "Avg f", "Median f", "Max f", "Std f"]
        
        # Sort by best_f
        sorted_funcs = sorted(results.items(), key=lambda x: x[1]['best_f'])
        
        for rank, (func_name, result) in enumerate(sorted_funcs, 1):
            table_rows.append([
                rank,
                func_name,
                f"{result['best_f']:.4e}",
                f"{result['avg_f']:.4e}",
                f"{result['median_f']:.4e}",
                f"{result['max_f']:.4e}",
                f"{result['std_f']:.4e}",
            ])
        
        tables[algo] = tabulate(table_rows, headers=headers, tablefmt="grid")
    
    return tables


def generate_convergence_plots(all_results, output_dir="."):
    """Generate convergence curve plots for each algorithm."""
    algorithms = list(all_results.keys())
    
    # Create a combined plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, algo in enumerate(algorithms[:4]):  # Max 4 algorithms
        ax = axes[idx]
        for func_name, result in all_results[algo].items():
            if 'convergence_history' in result and result['convergence_history']:
                ax.semilogy(result['convergence_history'], label=func_name, linewidth=1)
        
        ax.set_title(f"{algo} Convergence Curves (D={NUM_DIMENSIONS})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Cost (Log Scale)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, which="both", ls="-", alpha=0.5)
    
    # Hide unused subplots
    for idx in range(len(algorithms), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = Path(output_dir) / "convergence_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence plots saved to {output_file}")


def save_results(all_results, output_file="output.txt"):
    """Save all results to a text file."""
    comparison_table = generate_comparison_table(all_results)
    detailed_tables = generate_detailed_tables(all_results)
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("UNIFIED ALGORITHM COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Runs per function: {NUM_RUNS}\n")
        f.write(f"  Dimensions: {NUM_DIMENSIONS}\n")
        f.write("  Seeds: Each algorithm uses its own BASE_SEED (see algorithm folders)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ALGORITHM COMPARISON (Best Values)\n")
        f.write("=" * 80 + "\n\n")
        f.write(comparison_table)
        f.write("\n\n")
        
        for algo, table in detailed_tables.items():
            f.write("=" * 80 + "\n")
            f.write(f"{algo} DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(table)
            f.write("\n\n")
    
    print(f"Results saved to {output_file}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting Unified Algorithm Comparison...\n")
    
    # Run all algorithms
    all_results = run_all_algorithms()
    
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    
    # Generate comparison table
    comparison_table = generate_comparison_table(all_results)
    print("\n" + comparison_table + "\n")
    
    # Save results
    save_results(all_results)
    
    # Generate plots
    generate_convergence_plots(all_results)
    
    print("\n✅ All done!")
