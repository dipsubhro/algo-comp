from tabu import tabu_search
import numpy as np
BASE_SEED = 954777839

def run_tabu(fn, num_runs=25, neighbors=10, tenure=5, max_iter=1000, bounds=(-5, 5), dims=5):
    best_f = float('inf')
    best_x = None
    all_f = []
    all_histories = []
    seed = BASE_SEED
    for run in range(num_runs):
        np.random.seed(seed)
        x0 = np.random.uniform(bounds[0], bounds[1], size=dims)
        x, f, _, _, _, history = tabu_search(fn, x0, tenure=tenure, max_iter=max_iter, bounds=bounds, neighbors_size=neighbors)
        all_f.append(f)
        all_histories.append(history)
        if f < best_f:
            best_f = f
            best_x = x
        seed += 12345
    max_len = max((len(h) for h in all_histories))
    padded_histories = []
    for h in all_histories:
        if len(h) < max_len:
            padded = h + [h[-1]] * (max_len - len(h))
        else:
            padded = h
        padded_histories.append(padded)
    avg_history = np.mean(padded_histories, axis=0).tolist()
    return {'best_x': best_x, 'best_f': best_f, 'avg_f': np.mean(all_f), 'median_f': np.median(all_f), 'max_f': np.max(all_f), 'std_f': np.std(all_f), 'convergence_history': avg_history}