import statistics
import numpy as np
from collections import deque

def tabu_search(func, x0, tenure=7, max_iter=100, bounds=None, neighbors_size=20):
    num_dimensions = len(x0)
    if bounds:
        search_range = bounds[1] - bounds[0]
        lower_bound, upper_bound = (bounds[0], bounds[1])
    else:
        search_range = 10.0
        lower_bound, upper_bound = (-5, 5)
    initial_step_ratio = 0.15
    min_step_ratio = 0.001
    step_decay = 0.998
    min_dims_to_perturb = 1
    max_dims_to_perturb = max(2, num_dimensions // 2)
    stagnation_threshold = int(max_iter * 0.05)
    diversification_count = 0
    max_diversifications = 5
    current_solution = np.array(x0, dtype=float)
    best_solution = current_solution.copy()
    best_objective_value = func(current_solution)
    tabu_positions = deque(maxlen=tenure * 3)
    min_tenure = max(2, tenure // 2)
    max_tenure = tenure * 2
    current_tenure = tenure
    all_objective_values = []
    best_history = [best_objective_value]
    iterations_without_improvement = 0
    current_step_ratio = initial_step_ratio
    intensification_mode = False
    intensification_iters = 0
    max_intensification_iters = int(max_iter * 0.1)
    for iteration in range(max_iter):
        neighbors = []
        if intensification_mode:
            step_size = search_range * current_step_ratio * 0.3
        else:
            step_size = search_range * current_step_ratio
        for _ in range(neighbors_size):
            new_solution = current_solution.copy()
            if intensification_mode:
                num_perturb = 1
            else:
                num_perturb = np.random.randint(min_dims_to_perturb, max_dims_to_perturb + 1)
            dims_to_perturb = np.random.choice(num_dimensions, size=min(num_perturb, num_dimensions), replace=False)
            for dim_idx in dims_to_perturb:
                perturbation = np.random.normal(0, step_size / 2)
                new_solution[dim_idx] += perturbation
                new_solution[dim_idx] = np.clip(new_solution[dim_idx], lower_bound, upper_bound)
            new_objective_value = func(new_solution)
            all_objective_values.append(new_objective_value)
            position_hash = hash(tuple(np.round(new_solution, 4)))
            neighbors.append((new_solution, new_objective_value, position_hash))
        neighbors.sort(key=lambda t: t[1])
        best_neighbor = None
        best_neighbor_value = None
        best_neighbor_hash = None
        for neighbor_solution, neighbor_value, pos_hash in neighbors:
            if neighbor_value < best_objective_value:
                best_neighbor = neighbor_solution
                best_neighbor_value = neighbor_value
                best_neighbor_hash = pos_hash
                break
            if pos_hash not in tabu_positions:
                best_neighbor = neighbor_solution
                best_neighbor_value = neighbor_value
                best_neighbor_hash = pos_hash
                break
        if best_neighbor is None:
            best_neighbor = neighbors[0][0]
            best_neighbor_value = neighbors[0][1]
            best_neighbor_hash = neighbors[0][2]
        current_solution = best_neighbor
        current_objective_value = best_neighbor_value
        tabu_positions.append(best_neighbor_hash)
        if current_objective_value < best_objective_value:
            best_solution = current_solution.copy()
            best_objective_value = current_objective_value
            iterations_without_improvement = 0
            if not intensification_mode and iteration > max_iter * 0.3:
                intensification_mode = True
                intensification_iters = 0
            current_tenure = max(min_tenure, current_tenure - 1)
        else:
            iterations_without_improvement += 1
            if iterations_without_improvement > stagnation_threshold // 2:
                current_tenure = min(max_tenure, current_tenure + 1)
        if intensification_mode:
            intensification_iters += 1
            if intensification_iters >= max_intensification_iters:
                intensification_mode = False
                intensification_iters = 0
        if iterations_without_improvement >= stagnation_threshold:
            if diversification_count < max_diversifications:
                restart_perturbation = search_range * 0.2
                current_solution = best_solution + np.random.uniform(-restart_perturbation, restart_perturbation, size=num_dimensions)
                current_solution = np.clip(current_solution, lower_bound, upper_bound)
                current_step_ratio = initial_step_ratio * 0.7
                for _ in range(len(tabu_positions) // 2):
                    if tabu_positions:
                        tabu_positions.popleft()
                iterations_without_improvement = 0
                diversification_count += 1
                intensification_mode = False
        current_step_ratio = max(min_step_ratio, current_step_ratio * step_decay)
        best_history.append(best_objective_value)
    avg_f = sum(all_objective_values) / len(all_objective_values) if all_objective_values else 0
    median_f = statistics.median(all_objective_values) if all_objective_values else 0
    max_f = max(all_objective_values) if all_objective_values else 0
    return (list(best_solution), best_objective_value, avg_f, median_f, max_f, best_history)