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
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

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

# Try to import GA (may have different structure)
try:
    from ga_algorithm import run_ga, run_multiple_experiments as run_ga_multi, calculate_statistics
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False
    print("Warning: GA module not available for unified comparison")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_RUNS = 25
NUM_DIMENSIONS = 5
BASE_SEED = 12345

# Algorithm-specific configurations
PSO_CONFIG = {
    'n_particles': 40,
    'max_iterations': 1000,
}

SA_CONFIG = {
    'max_iterations': 1500,
    'initial_temp': 1000.0,
}

TABU_CONFIG = {
    'neighbors': 40,
    'tenure': 8,
    'max_iter': 2000,
}

GA_CONFIG = {
    'num_runs': NUM_RUNS,
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

# Step sizes for SA (tuned per function based on search domain)
SA_STEP_SIZES = {
    "Sphere": 0.5,
    "Sum_of_Squares": 1.0,
    "Sum_of_Diff_Powers": 0.1,
    "Step": 5.0,
    "Brown": 0.3,
    "Zakharov": 0.8,
    "Dixon_Price": 1.0,
    "Schumer_Steiglitz": 1.0,
    "Csendes": 0.1,
    "Sixth_Power": 0.1,
    "Powell": 0.5,
    "Quartic": 0.1,
    "Rotated_Hyper_Ellipsoid": 3.0,
    "Discus": 5.0,
    "Exponential": 0.1,
}


# ==============================================================================
# ALGORITHM RUNNERS
# ==============================================================================

def run_pso_experiment(args):
    """Run PSO on a single function."""
    name, fn, bounds, dims = args
    result = run_pso(
        fn, bounds,
        n_runs=NUM_RUNS,
        n_particles=PSO_CONFIG['n_particles'],
        n_dimensions=dims,
        max_iterations=PSO_CONFIG['max_iterations'],
        base_seed=BASE_SEED
    )
    return ("PSO", name, result)


def run_sa_experiment(args):
    """Run SA on a single function."""
    name, fn, bounds, dims = args
    step_size = SA_STEP_SIZES.get(name, 0.5)
    result = run_sa(
        fn, bounds,
        n_runs=NUM_RUNS,
        n_dimensions=dims,
        max_iterations=SA_CONFIG['max_iterations'],
        initial_temp=SA_CONFIG['initial_temp'],
        step_size=step_size,
        base_seed=BASE_SEED
    )
    return ("SA", name, result)


def run_tabu_experiment(args):
    """Run Tabu Search on a single function."""
    name, fn, bounds, dims = args
    result = run_tabu(
        fn,
        num_runs=NUM_RUNS,
        neighbors=TABU_CONFIG['neighbors'],
        tenure=TABU_CONFIG['tenure'],
        max_iter=TABU_CONFIG['max_iter'],
        bounds=bounds,
        dims=dims
    )
    return ("Tabu", name, result)


def run_ga_experiment(args):
    """Run GA on a single function."""
    if not GA_AVAILABLE:
        return None
    # GA has its own function mapping, need to use function name
    name, fn, bounds, dims = args
    # GA uses its internal benchmark_functions dict
    try:
        results, history = run_ga_multi(name, GA_CONFIG['num_runs'])
        stats = calculate_statistics(results)
        return ("GA", name, {
            'best_f': stats['min'],
            'avg_f': stats['mean'],
            'median_f': stats['median'],
            'max_f': stats['max'],
            'std_f': stats['std'],
            'convergence_history': history if history else [],
            'best_x': None  # GA doesn't return this in the same way
        })
    except Exception as e:
        print(f"GA failed for {name}: {e}")
        return None


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_all_algorithms():
    """Run all algorithms on all benchmark functions."""
    max_workers = max(1, os.cpu_count() - 2)
    
    print("=" * 80)
    print("UNIFIED ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"Algorithms: PSO, SA, Tabu Search" + (", GA" if GA_AVAILABLE else ""))
    print(f"Functions: {len(BENCHMARK_FUNCTIONS)}")
    print(f"Runs per function: {NUM_RUNS}")
    print(f"Dimensions: {NUM_DIMENSIONS}")
    print(f"Workers: {max_workers}")
    print("=" * 80)
    print()
    
    all_results = {}  # {algo: {func_name: result}}
    
    # Run PSO
    print("Running PSO...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pso_results = list(executor.map(run_pso_experiment, BENCHMARK_FUNCTIONS))
    all_results["PSO"] = {name: result for algo, name, result in pso_results}
    print("  PSO completed ✓")
    
    # Run SA
    print("Running SA...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sa_results = list(executor.map(run_sa_experiment, BENCHMARK_FUNCTIONS))
    all_results["SA"] = {name: result for algo, name, result in sa_results}
    print("  SA completed ✓")
    
    # Run Tabu Search
    print("Running Tabu Search...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tabu_results = list(executor.map(run_tabu_experiment, BENCHMARK_FUNCTIONS))
    all_results["Tabu"] = {name: result for algo, name, result in tabu_results}
    print("  Tabu Search completed ✓")
    
    # Optionally run GA
    if GA_AVAILABLE:
        print("Running GA...")
        ga_results_raw = []
        for func_data in BENCHMARK_FUNCTIONS:
            result = run_ga_experiment(func_data)
            if result:
                ga_results_raw.append(result)
        if ga_results_raw:
            all_results["GA"] = {name: result for algo, name, result in ga_results_raw}
        print("  GA completed ✓")
    
    return all_results


def generate_comparison_table(all_results):
    """Generate a unified comparison table."""
    algorithms = list(all_results.keys())
    func_names = [f[0] for f in BENCHMARK_FUNCTIONS]
    
    # Table 1: Best values per function per algorithm
    table_rows = []
    headers = ["Function"] + [f"{algo} Best" for algo in algorithms]
    
    for func_name in func_names:
        row = [func_name]
        for algo in algorithms:
            if func_name in all_results[algo]:
                best_f = all_results[algo][func_name]['best_f']
                row.append(f"{best_f:.4e}")
            else:
                row.append("N/A")
        table_rows.append(row)
    
    # Add ranking row
    ranking_row = ["RANK (Total Wins)"]
    for algo in algorithms:
        wins = 0
        for func_name in func_names:
            best_vals = []
            for a in algorithms:
                if func_name in all_results[a]:
                    best_vals.append((a, all_results[a][func_name]['best_f']))
            if best_vals:
                winner = min(best_vals, key=lambda x: x[1])[0]
                if winner == algo:
                    wins += 1
        ranking_row.append(str(wins))
    table_rows.append(ranking_row)
    
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
        f.write(f"  Base seed: {BASE_SEED}\n\n")
        
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
