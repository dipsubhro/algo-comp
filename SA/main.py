"""
Main experiment file for Simulated Annealing.
Runs all benchmark functions concurrently and outputs results.
"""

import sys
import os

# Add parent directory to path for importing shared func.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sa_algorithm import run_sa, SAConfig
from func import (
    sphere, sum_of_squares, sum_of_different_powers, step, brown,
    zakharov, dixon_price, schumer_steiglitz, csendes, sixth_power,
    powell, quartic, rotated_hyper_ellipsoid, discus, exponential
)
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt


# Configuration
NUM_RUNS = 25
NUM_DIMENSIONS = 5
MAX_ITERATIONS = 1500
INITIAL_TEMP = 1000.0
STEP_SIZE = 0.5
BASE_SEED = 84748


# Define all experiments: (name, fn, bounds, dims, step_size)
# Step size can be tuned per function based on search domain
experiments = [
    ("Sphere", sphere, (-5.12, 5.12), NUM_DIMENSIONS, 0.5),
    ("Sum_of_Squares", sum_of_squares, (-10, 10), NUM_DIMENSIONS, 1.0),
    ("Sum_of_Different_Powers", sum_of_different_powers, (-1, 1), NUM_DIMENSIONS, 0.1),
    ("Step", step, (-100, 100), NUM_DIMENSIONS, 5.0),
    ("Brown", brown, (-1, 4), NUM_DIMENSIONS, 0.3),
    ("Zakharov", zakharov, (-5, 10), NUM_DIMENSIONS, 0.8),
    ("Dixon_Price", dixon_price, (-10, 10), NUM_DIMENSIONS, 1.0),
    ("Schumer_Steiglitz", schumer_steiglitz, (-100, 100), NUM_DIMENSIONS, 5.0),
    ("Csendes", csendes, (-1, 1), NUM_DIMENSIONS, 0.1),
    ("Sixth_Power", sixth_power, (-1, 1), NUM_DIMENSIONS, 0.1),
    ("Powell", powell, (-4, 5), 4, 0.5),  # Powell works best with dims divisible by 4
    ("Quartic", quartic, (-1.28, 1.28), NUM_DIMENSIONS, 0.1),
    ("Rotated_Hyper_Ellipsoid", rotated_hyper_ellipsoid, (-65.536, 65.536), NUM_DIMENSIONS, 3.0),
    ("Discus", discus, (-100, 100), NUM_DIMENSIONS, 5.0),
    ("Exponential", exponential, (-1, 1), NUM_DIMENSIONS, 0.1),
]


def run_experiment(args):
    """Run a single experiment (called by each worker)."""
    name, fn, bounds, dims, step_size = args
    print(f"Running {name}...")
    result = run_sa(
        fn, bounds,
        n_runs=NUM_RUNS,
        n_dimensions=dims,
        max_iterations=MAX_ITERATIONS,
        initial_temp=INITIAL_TEMP,
        step_size=step_size,
        base_seed=BASE_SEED
    )
    return (name, result, bounds, dims, step_size)


def visualize_convergence(results, output_file="convergence_curves.png"):
    """Create convergence curve plot."""
    plt.figure(figsize=(12, 7))
    
    for name, result, _, _, _ in results:
        if 'avg_history' in result:
            plt.semilogy(result['avg_history'], label=name)
    
    plt.title(f"SA Convergence Curves (D={NUM_DIMENSIONS}, T₀={INITIAL_TEMP})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost (Log Scale)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence curves saved to {output_file}")


if __name__ == "__main__":
    # Run all experiments concurrently
    max_workers = max(1, os.cpu_count() - 2)
    
    print("=" * 70)
    print("SIMULATED ANNEALING BENCHMARK EXPERIMENTS")
    print("=" * 70)
    print(f"Runs per function: {NUM_RUNS}")
    print(f"Dimensions: {NUM_DIMENSIONS}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Initial temperature: {INITIAL_TEMP}")
    print(f"Workers: {max_workers}")
    print("=" * 70)
    print()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_experiment, experiments))
    
    # Sort results by best_f
    sorted_results = sorted(results, key=lambda x: x[1]['best_f'])
    rank_map = {name: rank + 1 for rank, (name, *_) in enumerate(sorted_results)}
    
    # Build table rows
    table_rows = []
    for name, result, bounds, dims, step_size in sorted_results:
        best_x_str = "[" + ", ".join(f"{x:.4f}" for x in result['best_x']) + "]"
        table_rows.append([
            rank_map[name],
            name,
            NUM_RUNS,
            f"{bounds}",
            dims,
            step_size,
            f"{result['best_f']:.4e}",
            f"{result['avg_f']:.4e}",
            f"{result['median_f']:.4e}",
            f"{result['max_f']:.4e}",
            f"{result['std_f']:.4e}",
            best_x_str
        ])
    
    headers = [
        "Rank", "Function", "Runs", "Bounds", "Dims", "Step",
        "Best f", "Avg f", "Median f", "Max f", "Std f", "Best x"
    ]
    
    table = tabulate(table_rows, headers=headers, tablefmt="grid")
    
    # Write to file
    with open("output.txt", "w") as f:
        f.write("Simulated Annealing Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Runs: {NUM_RUNS}\n")
        f.write(f"  Dimensions: {NUM_DIMENSIONS}\n")
        f.write(f"  Max Iterations: {MAX_ITERATIONS}\n")
        f.write(f"  Initial Temperature: {INITIAL_TEMP}\n\n")
        f.write(table)
        f.write("\n")
    
    print("\n" + "=" * 70)
    print("SA PERFORMANCE SUMMARY")
    print("=" * 70)
    print(table)
    print("\nDone! Results saved to output.txt")
    
    # Generate convergence visualization
    visualize_convergence(results)
