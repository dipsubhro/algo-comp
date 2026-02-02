import numpy as np
from typing import Callable, Dict, Any, Tuple, List
BASE_SEED = 84748

class SAConfig:

    def __init__(self, n_dimensions: int=5, max_iterations: int=1500, initial_temp: float=1000.0, min_temp: float=1e-06, cooling_rate: float=0.95, initial_acceptance: float=0.9, final_acceptance: float=0.01, markov_length: int=50, step_size: float=0.5):
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.initial_acceptance = initial_acceptance
        self.final_acceptance = final_acceptance
        self.markov_length = markov_length
        self.step_size = step_size

    def to_dict(self) -> Dict[str, Any]:
        return {'n_dimensions': self.n_dimensions, 'max_iterations': self.max_iterations, 'initial_temp': self.initial_temp, 'min_temp': self.min_temp, 'cooling_rate': self.cooling_rate, 'initial_acceptance': self.initial_acceptance, 'final_acceptance': self.final_acceptance, 'markov_length': self.markov_length, 'step_size': self.step_size}

class SimulatedAnnealing:

    def __init__(self, config: SAConfig=None):
        self.config = config or SAConfig()

    def optimize(self, objective_function: Callable, bounds: Tuple[float, float], seed: int=None) -> Tuple[float, np.ndarray, List[float]]:
        cfg = self.config
        x_min, x_max = bounds
        if seed is not None:
            np.random.seed(seed)
        S = np.random.uniform(x_min, x_max, cfg.n_dimensions)
        S_star = S.copy()
        S_star_value = objective_function(S_star)
        T = cfg.initial_temp
        history = []
        for iteration in range(cfg.max_iterations):
            accepted = 0
            for _ in range(cfg.markov_length):
                S_prime = S + np.random.uniform(-cfg.step_size, cfg.step_size, cfg.n_dimensions)
                S_prime = np.clip(S_prime, x_min, x_max)
                current_energy = objective_function(S)
                new_energy = objective_function(S_prime)
                delta = new_energy - current_energy
                if delta <= 0 or np.random.rand() < np.exp(-delta / T):
                    S = S_prime
                    accepted += 1
                    if new_energy < S_star_value:
                        S_star = S.copy()
                        S_star_value = new_energy
            history.append(S_star_value)
            acc_ratio = accepted / cfg.markov_length
            if acc_ratio > cfg.initial_acceptance:
                T = T / 2
            else:
                T = cfg.cooling_rate * T
            if T < cfg.min_temp:
                break
        return (S_star_value, S_star, history)

def run_sa(objective_function: Callable, bounds: Tuple[float, float], n_runs: int=25, n_dimensions: int=5, max_iterations: int=1500, initial_temp: float=1000.0, step_size: float=0.5, base_seed: int=None) -> Dict[str, Any]:
    if base_seed is None:
        base_seed = BASE_SEED
    config = SAConfig(n_dimensions=n_dimensions, max_iterations=max_iterations, initial_temp=initial_temp, step_size=step_size)
    sa = SimulatedAnnealing(config)
    final_values = []
    all_histories = []
    best_position = None
    best_value = float('inf')
    for run in range(n_runs):
        seed = base_seed + run
        value, position, history = sa.optimize(objective_function, bounds, seed)
        final_values.append(value)
        all_histories.append(history)
        if value < best_value:
            best_value = value
            best_position = position
    values_array = np.array(final_values)
    max_len = max((len(h) for h in all_histories))
    padded_histories = []
    for h in all_histories:
        if len(h) < max_len:
            h = h + [h[-1]] * (max_len - len(h))
        padded_histories.append(h)
    avg_history = np.mean(padded_histories, axis=0)
    return {'best_f': np.min(values_array), 'avg_f': np.mean(values_array), 'median_f': np.median(values_array), 'max_f': np.max(values_array), 'std_f': np.std(values_array), 'best_x': best_position, 'convergence_history': avg_history.tolist()}
if __name__ == '__main__':

    def sphere(x):
        return sum((i ** 2 for i in x))
    result = run_sa(sphere, (-5.12, 5.12), n_runs=5, max_iterations=100)
    print(f'Best: {result['best_f']:.6e}')
    print(f'Mean: {result['avg_f']:.6e}')
    print(f'Std:  {result['std_f']:.6e}')