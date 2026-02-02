from run_tabu import run_tabu
from func import sphere, sum_of_squares, sum_of_different_powers, step, brown, zakharov, dixon_price, schumer_steiglitz, csendes, sixth_power, powell, quartic, rotated_hyper_ellipsoid, discus, exponential
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate
from visualize import visualize_results, create_unigraph, create_convergence_curves

def run_experiment(args):
    name, fn, neighbors, tenure, max_iter, bounds, dims = args
    print(f'Running {name}...')
    result = run_tabu(fn, NUM_RUNS, neighbors, tenure, max_iter, bounds, dims)
    return (name, result, NUM_RUNS, neighbors, tenure, max_iter, bounds, dims)
NUM_RUNS = 25
STANDARD_DIMS = 5
experiments = [('Sphere', sphere, 40, 8, 2000, (-5.12, 5.12), STANDARD_DIMS), ('Sum_of_Squares', sum_of_squares, 40, 8, 2000, (-10, 10), STANDARD_DIMS), ('Sum_of_Different_Powers', sum_of_different_powers, 40, 8, 2000, (-1, 1), STANDARD_DIMS), ('Step', step, 60, 12, 2000, (-5.12, 5.12), STANDARD_DIMS), ('Brown', brown, 40, 8, 2000, (-1, 4), STANDARD_DIMS), ('Zakharov', zakharov, 50, 10, 2500, (-5, 10), STANDARD_DIMS), ('Dixon_Price', dixon_price, 60, 12, 3000, (-10, 10), STANDARD_DIMS), ('Schumer_Steiglitz', schumer_steiglitz, 40, 8, 2000, (-10, 10), STANDARD_DIMS), ('Csendes', csendes, 40, 8, 2000, (-1, 1), STANDARD_DIMS), ('Sixth_Power', sixth_power, 40, 8, 2000, (-1, 1), STANDARD_DIMS), ('Powell', powell, 50, 10, 2500, (-4, 5), 4), ('Quartic', quartic, 35, 7, 1500, (-1.28, 1.28), STANDARD_DIMS), ('Rotated_Hyper_Ellipsoid', rotated_hyper_ellipsoid, 50, 10, 2500, (-30, 30), STANDARD_DIMS), ('Discus', discus, 60, 12, 3000, (-10, 10), STANDARD_DIMS), ('Exponential', exponential, 35, 7, 1500, (-1, 1), STANDARD_DIMS)]
if __name__ == '__main__':
    import os
    max_workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_experiment, experiments))
    sorted_results = sorted(results, key=lambda x: x[1]['best_f'])
    rank_map = {name: rank + 1 for rank, (name, *_) in enumerate(sorted_results)}
    table_rows = []
    for name, result, num_runs, neighbors, tenure, max_iter, bounds, dims in sorted_results:
        best_x_str = '[' + ', '.join((f'{x:.4f}' for x in result['best_x'])) + ']'
        table_rows.append([rank_map[name], name, num_runs, neighbors, tenure, max_iter, f'{bounds}', dims, f'{result['best_f']:.4e}', f'{result['avg_f']:.4e}', f'{result['median_f']:.4e}', f'{result['max_f']:.4e}', f'{result['std_f']:.4e}', best_x_str])
    headers = ['Rank', 'Function', 'Runs', 'Neighbors', 'Tenure', 'MaxIter', 'Bounds', 'Dims', 'Best f', 'Avg f', 'Median f', 'Max f', 'Std f', 'Best x']
    table = tabulate(table_rows, headers=headers, tablefmt='grid')
    with open('output.txt', 'w') as f:
        f.write('Tabu Search Results\n')
        f.write('=' * 60 + '\n\n')
        f.write(table)
        f.write('\n')
    print('Done! Results saved to output.txt')
    visualize_results(results)
    create_unigraph(results)
    create_convergence_curves(results)