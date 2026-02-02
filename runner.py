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
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent

sys.path.insert(0, str(ROOT / 'PSO'))
sys.path.insert(0, str(ROOT / 'SA'))
sys.path.insert(0, str(ROOT / 'TA'))
sys.path.insert(0, str(ROOT / 'GA'))
sys.path.insert(0, str(ROOT))

from func import sphere, csendes, sum_of_different_powers, step, powell, dixon_price, schumer_steiglitz, sixth_power, quartic, exponential
from func import rastrigin, ackley, griewank, schwefel, levy, styblinski_tang, alpine, schaffer_f6, rosenbrock, drop_wave
from pso_algorithm import run_pso
from sa_algorithm import run_sa
from run_tabu import run_tabu

try:
    from ga_algorithm import run_ga
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False
    print('Warning: GA module not available for unified comparison')

NUM_RUNS = 30
NUM_DIMENSIONS = 5

from pso_config import PSO_CONFIG, PSO_CONFIGS
from sa_config import SA_CONFIG, SA_CONFIGS
from tabu_config import TABU_CONFIG, TABU_CONFIGS
from ga_config import GA_CONFIG, GA_CONFIGS

BENCHMARK_FUNCTIONS = [
    ('Sphere', sphere, (-5.12, 5.12), NUM_DIMENSIONS),
    ('Csendes', csendes, (-1, 1), NUM_DIMENSIONS),
    ('Sum_of_Diff_Powers', sum_of_different_powers, (-1, 1), NUM_DIMENSIONS),
    ('Step', step, (-100, 100), NUM_DIMENSIONS),
    ('Powell', powell, (-4, 5), NUM_DIMENSIONS),
    ('Dixon_Price', dixon_price, (-10, 10), NUM_DIMENSIONS),
    ('Schumer_Steiglitz', schumer_steiglitz, (-10, 10), NUM_DIMENSIONS),
    ('Sixth_Power', sixth_power, (-1, 1), NUM_DIMENSIONS),
    ('Quartic', quartic, (-1.28, 1.28), NUM_DIMENSIONS),
    ('Exponential', exponential, (-1, 1), NUM_DIMENSIONS),
    ('Rastrigin', rastrigin, (-5.12, 5.12), NUM_DIMENSIONS),
    ('Ackley', ackley, (-32.768, 32.768), NUM_DIMENSIONS),
    ('Griewank', griewank, (-600, 600), NUM_DIMENSIONS),
    ('Schwefel', schwefel, (-500, 500), NUM_DIMENSIONS),
    ('Levy', levy, (-10, 10), NUM_DIMENSIONS),
    ('Styblinski_Tang', styblinski_tang, (-5, 5), NUM_DIMENSIONS),
    ('Alpine', alpine, (-10, 10), NUM_DIMENSIONS),
    ('Schaffer_F6', schaffer_f6, (-100, 100), NUM_DIMENSIONS),
    ('Rosenbrock', rosenbrock, (-5, 10), NUM_DIMENSIONS),
    ('Drop_Wave', drop_wave, (-5.12, 5.12), NUM_DIMENSIONS)
]

def run_pso_experiment(args):
    name, fn, bounds, dims = args
    start = time.time()
    logger.info(f'[PSO] Starting {name}...')
    config = PSO_CONFIGS.get(name, PSO_CONFIG)
    result = run_pso(fn, bounds, n_runs=NUM_RUNS, n_particles=config['n_particles'], n_dimensions=dims, max_iterations=config['max_iterations'])
    elapsed = time.time() - start
    logger.info(f'[PSO] {name} completed in {elapsed:.2f}s | Best: {result['best_f']:.4e}')
    return ('PSO', name, result)


def run_sa_experiment(args):
    name, fn, bounds, dims = args
    start = time.time()
    logger.info(f'[SA] Starting {name}...')
    config = SA_CONFIGS.get(name, SA_CONFIG)
    result = run_sa(fn, bounds, n_runs=NUM_RUNS, n_dimensions=dims, max_iterations=config['max_iterations'], initial_temp=config['initial_temp'], step_size=config['step_size'])
    elapsed = time.time() - start
    logger.info(f'[SA] {name} completed in {elapsed:.2f}s | Best: {result['best_f']:.4e}')
    return ('SA', name, result)


def run_tabu_experiment(args):
    name, fn, bounds, dims = args
    start = time.time()
    logger.info(f'[Tabu] Starting {name}...')
    config = TABU_CONFIGS.get(name, TABU_CONFIG)
    result = run_tabu(fn, num_runs=NUM_RUNS, neighbors=config['neighbors'], tenure=config['tenure'], max_iter=config['max_iter'], bounds=bounds, dims=dims)
    elapsed = time.time() - start
    logger.info(f'[Tabu] {name} completed in {elapsed:.2f}s | Best: {result['best_f']:.4e}')
    return ('Tabu', name, result)


def run_ga_experiment(args):
    if not GA_AVAILABLE:
        return None
    name, fn, bounds, dims = args
    start = time.time()
    logger.info(f'[GA] Starting {name}...')
    config = GA_CONFIGS.get(name, GA_CONFIG)
    try:
        result = run_ga(fn, bounds, n_runs=NUM_RUNS, n_dimensions=dims, pop_size=config['pop_size'], num_generations=config['num_generations'], crossover_rate=config['crossover_rate'], mutation_rate=config['mutation_rate'], elite_size=config['elite_size'])
        elapsed = time.time() - start
        logger.info(f'[GA] {name} completed in {elapsed:.2f}s | Best: {result['best_f']:.4e}')
        return ('GA', name, result)
    except Exception as e:
        logger.error(f'[GA] {name} failed: {e}')
        return None


def run_all_algorithms():
    max_workers = max(1, os.cpu_count() - 2)
    total_start = time.time()
    logger.info('=' * 60)
    logger.info('UNIFIED ALGORITHM COMPARISON')
    logger.info('=' * 60)
    logger.info(f'Algorithms: PSO' + (', GA' if GA_AVAILABLE else '') + ', Tabu Search, SA')
    logger.info(f'Functions: {len(BENCHMARK_FUNCTIONS)}')
    logger.info(f'Runs per function: {NUM_RUNS}')
    logger.info(f'Dimensions: {NUM_DIMENSIONS}')
    logger.info(f'Workers: {max_workers}')
    logger.info('=' * 60)
    all_results = {}
    logger.info('')

    logger.info('▶ Starting PSO experiments...')
    pso_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pso_results = list(executor.map(run_pso_experiment, BENCHMARK_FUNCTIONS))
    all_results['PSO'] = {name: result for algo, name, result in pso_results}
    logger.info(f'✓ PSO completed in {time.time() - pso_start:.2f}s')

    if GA_AVAILABLE:
        logger.info('')
        logger.info('▶ Starting GA experiments...')
        ga_start = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            ga_results_raw = list(executor.map(run_ga_experiment, BENCHMARK_FUNCTIONS))
        ga_results_filtered = [r for r in ga_results_raw if r is not None]
        if ga_results_filtered:
            all_results['GA'] = {name: result for algo, name, result in ga_results_filtered}
        logger.info(f'✓ GA completed in {time.time() - ga_start:.2f}s')

    logger.info('')
    logger.info('▶ Starting Tabu Search experiments...')
    tabu_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tabu_results = list(executor.map(run_tabu_experiment, BENCHMARK_FUNCTIONS))
    all_results['Tabu'] = {name: result for algo, name, result in tabu_results}
    logger.info(f'✓ Tabu Search completed in {time.time() - tabu_start:.2f}s')

    logger.info('')
    logger.info('▶ Starting SA experiments...')
    sa_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sa_results = list(executor.map(run_sa_experiment, BENCHMARK_FUNCTIONS))
    all_results['SA'] = {name: result for algo, name, result in sa_results}
    logger.info(f'✓ SA completed in {time.time() - sa_start:.2f}s')

    total_elapsed = time.time() - total_start
    logger.info('')
    logger.info(f'🏁 All algorithms completed in {total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)')
    return all_results


def generate_comparison_table(all_results):
    algorithm_order = ['PSO', 'GA', 'Tabu', 'SA']
    algorithms = [algo for algo in algorithm_order if algo in all_results]
    func_names = [f[0] for f in BENCHMARK_FUNCTIONS]
    table_rows = []
    headers = ['Function', 'Algorithm', 'Best', 'Average', 'Std Dev', 'Rank']
    for func_name in func_names:
        algo_best_values = []
        for algo in algorithms:
            if func_name in all_results[algo]:
                best_f = all_results[algo][func_name]['best_f']
                algo_best_values.append((algo, best_f))
        algo_best_values.sort(key=lambda x: x[1])
        rank_map = {algo: rank + 1 for rank, (algo, _) in enumerate(algo_best_values)}
        first_algo = True
        for algo in algorithms:
            if func_name in all_results[algo]:
                result = all_results[algo][func_name]
                best_f = result['best_f']
                avg_f = result['avg_f']
                std_f = result['std_f']
                rank = rank_map.get(algo, 'N/A')
                func_display = func_name if first_algo else ''
                table_rows.append([func_display, algo, f'{best_f:.2e}', f'{avg_f:.2e}', f'{std_f:.2e}', rank])
                first_algo = False
    return tabulate(table_rows, headers=headers, tablefmt='grid')


def generate_detailed_tables(all_results):
    tables = {}
    for algo, results in all_results.items():
        table_rows = []
        headers = ['Rank', 'Function', 'Best f', 'Avg f', 'Median f', 'Max f', 'Std f']
        sorted_funcs = sorted(results.items(), key=lambda x: x[1]['best_f'])
        for rank, (func_name, result) in enumerate(sorted_funcs, 1):
            table_rows.append([rank, func_name, f'{result['best_f']:.4e}', f'{result['avg_f']:.4e}', f'{result['median_f']:.4e}', f'{result['max_f']:.4e}', f'{result['std_f']:.4e}'])
        tables[algo] = tabulate(table_rows, headers=headers, tablefmt='grid')
    return tables


def generate_convergence_plots(all_results, output_dir='.'):
    algorithms = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, algo in enumerate(algorithms[:4]):
        ax = axes[idx]
        for func_name, result in all_results[algo].items():
            if 'convergence_history' in result and result['convergence_history']:
                ax.semilogy(result['convergence_history'], label=func_name, linewidth=1)
        ax.set_title(f'{algo} Convergence Curves (D={NUM_DIMENSIONS})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Cost (Log Scale)')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, which='both', ls='-', alpha=0.5)
    for idx in range(len(algorithms), 4):
        axes[idx].set_visible(False)
    plt.tight_layout()
    output_file = Path(output_dir) / 'convergence_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Convergence plots saved to {output_file}')


def save_results(all_results, output_file='output.txt'):
    comparison_table = generate_comparison_table(all_results)
    detailed_tables = generate_detailed_tables(all_results)
    with open(output_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('UNIFIED ALGORITHM COMPARISON RESULTS\n')
        f.write('=' * 80 + '\n\n')
        f.write('Configuration:\n')
        f.write(f'  Runs per function: {NUM_RUNS}\n')
        f.write(f'  Dimensions: {NUM_DIMENSIONS}\n')
        f.write('  Seeds: Each algorithm uses its own BASE_SEED (see algorithm folders)\n\n')
        f.write('=' * 80 + '\n')
        f.write('ALGORITHM COMPARISON (Best Values)\n')
        f.write('=' * 80 + '\n\n')
        f.write(comparison_table)
        f.write('\n\n')
        for algo, table in detailed_tables.items():
            f.write('=' * 80 + '\n')
            f.write(f'{algo} DETAILED RESULTS\n')
            f.write('=' * 80 + '\n\n')
            f.write(table)
            f.write('\n\n')
    print(f'Results saved to {output_file}')


if __name__ == '__main__':
    print('\n🚀 Starting Unified Algorithm Comparison...\n')
    all_results = run_all_algorithms()
    print('\n' + '=' * 80)
    print('GENERATING OUTPUTS')
    print('=' * 80)
    comparison_table = generate_comparison_table(all_results)
    print('\n' + comparison_table + '\n')
    save_results(all_results)
    generate_convergence_plots(all_results)
    print('\n✅ All done!')