import numpy as np
import random
from typing import Callable, Dict, Any, Tuple, List
BASE_SEED = 12345

class PSOConfig:

    def __init__(self, n_particles: int=40, n_dimensions: int=5, max_iterations: int=1000, w_start: float=0.9, w_end: float=0.4, c1: float=2.0, c2: float=2.0):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2

    def to_dict(self) -> Dict[str, Any]:
        return {'N': self.n_particles, 'D': self.n_dimensions, 'max_iterations': self.max_iterations, 'w_start': self.w_start, 'w_end': self.w_end, 'c1': self.c1, 'c2': self.c2}

class PSO:

    def __init__(self, config: PSOConfig=None):
        self.config = config or PSOConfig()

    def optimize(self, objective_function: Callable, bounds: Tuple[float, float], seed: int=None) -> Tuple[float, np.ndarray, List[float]]:
        cfg = self.config
        x_min, x_max = bounds
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        x = np.random.uniform(x_min, x_max, size=(cfg.n_particles, cfg.n_dimensions))
        v = np.zeros((cfg.n_particles, cfg.n_dimensions))
        pVal = np.array([objective_function(x[i]) for i in range(cfg.n_particles)])
        pBest_rep = [[x[i].copy()] for i in range(cfg.n_particles)]
        Gval = np.min(pVal)
        gBest_rep = [x[np.argmin(pVal)].copy()]
        history = []
        for t in range(cfg.max_iterations):
            w = cfg.w_start - (cfg.w_start - cfg.w_end) * t / cfg.max_iterations
            r1 = np.random.rand(cfg.n_particles, cfg.n_dimensions)
            r2 = np.random.rand(cfg.n_particles, cfg.n_dimensions)
            for i in range(cfg.n_particles):
                current_pBest = random.choice(pBest_rep[i])
                current_gBest = random.choice(gBest_rep)
                cognitive = cfg.c1 * r1[i] * (current_pBest - x[i])
                social = cfg.c2 * r2[i] * (current_gBest - x[i])
                v[i] = w * v[i] + cognitive + social
                x[i] = x[i] + v[i]
                x[i] = np.clip(x[i], x_min, x_max)
                current_val = objective_function(x[i])
                if current_val < pVal[i]:
                    pVal[i] = current_val
                    pBest_rep[i] = [x[i].copy()]
                elif current_val == pVal[i]:
                    if not any((np.array_equal(x[i], pos) for pos in pBest_rep[i])):
                        pBest_rep[i].append(x[i].copy())
                if current_val < Gval:
                    Gval = current_val
                    gBest_rep = [x[i].copy()]
                elif current_val == Gval:
                    if not any((np.array_equal(x[i], pos) for pos in gBest_rep)):
                        gBest_rep.append(x[i].copy())
            history.append(Gval)
        return (Gval, gBest_rep[0], history)

def run_pso(objective_function: Callable, bounds: Tuple[float, float], n_runs: int=30, n_particles: int=40, n_dimensions: int=5, max_iterations: int=1000, base_seed: int=None) -> Dict[str, Any]:
    if base_seed is None:
        base_seed = BASE_SEED
    config = PSOConfig(n_particles=n_particles, n_dimensions=n_dimensions, max_iterations=max_iterations)
    pso = PSO(config)
    final_values = []
    all_histories = []
    best_position = None
    best_value = float('inf')
    for run in range(n_runs):
        seed = base_seed + run
        value, position, history = pso.optimize(objective_function, bounds, seed)
        final_values.append(value)
        all_histories.append(history)
        if value < best_value:
            best_value = value
            best_position = position
    values_array = np.array(final_values)
    avg_history = np.mean(all_histories, axis=0)
    return {'best_f': np.min(values_array), 'avg_f': np.mean(values_array), 'median_f': np.median(values_array), 'max_f': np.max(values_array), 'std_f': np.std(values_array), 'best_x': best_position, 'convergence_history': avg_history.tolist()}
if __name__ == '__main__':

    def sphere(x):
        return sum((i ** 2 for i in x))
    result = run_pso(sphere, (-5.12, 5.12), n_runs=5, max_iterations=100)
    print(f'Best: {result['best_f']:.6e}')
    print(f'Mean: {result['avg_f']:.6e}')
    print(f'Std:  {result['std_f']:.6e}')