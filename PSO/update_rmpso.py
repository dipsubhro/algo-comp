import numpy as np
import sys
from tabulate import tabulate 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random
import math

# Global parameters
GLOBAL_PARAMS = {
    'N_RUNS': 30,
    'D': 5,
    'N': 40,
    'w': 0.7,
    'c1': 2.0,
    'c2': 2.0,
    'max_iterations': 1000
}

# --- 1. New 15 CEC-style Objective Functions ---
def sphere(x, p=None):
    return sum(i**2 for i in x)

def sum_of_squares(x, p=None):
    return sum((i + 1) * xi**2 for i, xi in enumerate(x))

def sum_of_different_powers(x, p=None):
    return sum(abs(xi) ** (i + 2) for i, xi in enumerate(x))

def step(x, p=None):
    return sum(int(i)**2 for i in x)

def brown(x, p=None):
    n = len(x)
    return sum((x[i]**2) ** (x[i+1]**2 + 1) + (x[i+1]**2) ** (x[i]**2 + 1) for i in range(n - 1))

def zakharov(x, p=None):
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(0.5 * (i + 1) * xi for i, xi in enumerate(x))
    return sum1 + sum2**2 + sum2**4

def dixon_price(x, p=None):
    n = len(x)
    term1 = (x[0] - 1)**2
    term2 = sum((i + 1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, n))
    return term1 + term2

def schumer_steiglitz(x, p=None):
    return sum(xi**4 for xi in x)

def csendes(x, p=None):
    return sum(xi**6 * (2 + math.sin(1 / (xi + 1e-10))) for xi in x)

def sixth_power(x, p=None):
    return sum(xi**6 for xi in x)

def powell(x, p=None):
    n = len(x)
    result = 0
    for i in range(0, n - 3, 4):
        result += (x[i] + 10*x[i+1])**2
        result += 5 * (x[i+2] - x[i+3])**2
        result += (x[i+1] - 2*x[i+2])**4
        result += 10 * (x[i] - x[i+3])**4
    return result

def quartic(x, p=None):
    return sum((i + 1) * xi**4 for i, xi in enumerate(x))

def rotated_hyper_ellipsoid(x, p=None):
    n = len(x)
    result = 0
    for i in range(n):
        result += sum(x[j]**2 for j in range(i + 1))
    return result

def discus(x, p=None):
    return 1e6 * x[0]**2 + sum(xi**2 for xi in x[1:])

def exponential(x, p=None):
    return -math.exp(-0.5 * sum(xi**2 for xi in x)) + 1

# --- 2. Function Configuration List ---
ALL_FUNCTION_CONFIGS = [
    {"id": 1, "name": "Sphere", "func": sphere, "min_range": -5.12, "max_range": 5.12, "params": {}},
    {"id": 2, "name": "Sum of Squares", "func": sum_of_squares, "min_range": -10.0, "max_range": 10.0, "params": {}},
    {"id": 3, "name": "Sum Diff Powers", "func": sum_of_different_powers, "min_range": -1.0, "max_range": 1.0, "params": {}},
    {"id": 4, "name": "Step", "func": step, "min_range": -100.0, "max_range": 100.0, "params": {}},
    {"id": 5, "name": "Brown", "func": brown, "min_range": -1.0, "max_range": 4.0, "params": {}},
    {"id": 6, "name": "Zakharov", "func": zakharov, "min_range": -5.0, "max_range": 10.0, "params": {}},
    {"id": 7, "name": "Dixon-Price", "func": dixon_price, "min_range": -10.0, "max_range": 10.0, "params": {}},
    {"id": 8, "name": "Schumer-Steiglitz", "func": schumer_steiglitz, "min_range": -1.21, "max_range": 1.21, "params": {}},
    {"id": 9, "name": "Csendes", "func": csendes, "min_range": -1.0, "max_range": 1.0, "params": {}},
    {"id": 10, "name": "Sixth Power", "func": sixth_power, "min_range": -10.0, "max_range": 10.0, "params": {}},
    {"id": 11, "name": "Powell", "func": powell, "min_range": -4.0, "max_range": 5.0, "params": {}},
    {"id": 12, "name": "Quartic", "func": quartic, "min_range": -1.28, "max_range": 1.28, "params": {}},
    {"id": 13, "name": "Rotated Hyper Ellipsoid", "func": rotated_hyper_ellipsoid, "min_range": -65.536, "max_range": 65.536, "params": {}},
    {"id": 14, "name": "Discus", "func": discus, "min_range": -5.12, "max_range": 5.12, "params": {}},
    {"id": 15, "name": "Exponential", "func": exponential, "min_range": -1.0, "max_range": 1.0, "params": {}},
]

# --- 3. Core RMPSO Run Function ---
def run_pso_once(objective_function, x_min_range, x_max_range, seed, pso_params, func_params):
    D = pso_params['D']
    N = pso_params['N']
    c1 = pso_params['c1']
    c2 = pso_params['c2']
    max_iterations = pso_params['max_iterations']

    np.random.seed(seed) 

    x = np.random.uniform(x_min_range, x_max_range, size=(N, D))
    v = np.zeros((N, D))

    pVal = np.array([objective_function(x[i], func_params) for i in range(N)])
    pBest_rep = [[x[i].copy()] for i in range(N)]
    
    Gval = np.min(pVal)
    gBest_rep = [x[np.argmin(pVal)].copy()]

    history = []

    for t in range(max_iterations):
        w_dynamic = 0.9 - ((0.9 - 0.4) * t / max_iterations)
        
        r1 = np.random.rand(N, D)
        r2 = np.random.rand(N, D)
        
        for i in range(N):
            current_pBest = random.choice(pBest_rep[i])
            current_gBest = random.choice(gBest_rep)

            cognitive_term = c1 * r1[i] * (current_pBest - x[i])
            social_term = c2 * r2[i] * (current_gBest - x[i])
            v[i] = w_dynamic * v[i] + cognitive_term + social_term
        
            x[i] = x[i] + v[i]
            x[i] = np.clip(x[i], x_min_range, x_max_range)

            current_val = objective_function(x[i], func_params)
        
            if current_val < pVal[i]:
                pVal[i] = current_val
                pBest_rep[i] = [x[i].copy()]
            elif current_val == pVal[i]:
                if not any(np.array_equal(x[i], pos) for pos in pBest_rep[i]):
                    pBest_rep[i].append(x[i].copy())
        
            if current_val < Gval:
                Gval = current_val
                gBest_rep = [x[i].copy()]
            elif current_val == Gval:
                if not any(np.array_equal(x[i], pos) for pos in gBest_rep):
                    gBest_rep.append(x[i].copy())
    
        history.append(Gval)
    
    return Gval, history

# --- 4. User Input ---
def get_user_input():
    print("\n--- PSO Configuration ---")
    p = GLOBAL_PARAMS.copy()
    try:
        p['N_RUNS'] = int(input(f"Number of runs [30]: ") or p['N_RUNS'])
        p['D'] = int(input(f"Dimensionality (D) [5]: ") or p['D'])
        p['N'] = int(input(f"Number of particles (N) [40]: ") or p['N'])
        p['max_iterations'] = int(input(f"Max iterations [1000]: ") or p['max_iterations'])
    except ValueError: print("Reverting to defaults.")
    
    print("\n--- Objective Function Selection ---")
    func_table_data = [[f['id'], f['name'], f['min_range'], f['max_range']] for f in ALL_FUNCTION_CONFIGS]
    print(tabulate(func_table_data, headers=["ID", "Name", "Min", "Max"], tablefmt="simple"))
    
    selection_str = input("\nEnter IDs (e.g., '1, 3' or '1-15'): ")
    selected_ids = set()
    try:
        for part in selection_str.split(','):
            if '-' in part.strip():
                start, end = map(int, part.split('-'))
                selected_ids.update(range(start, end + 1))
            else: selected_ids.add(int(part.strip()))
    except: 
        print("Invalid selection. Selecting all.")
        selected_ids = set(range(1, 16))
    
    selected_configs = [f.copy() for f in ALL_FUNCTION_CONFIGS if f['id'] in selected_ids]
    return p, selected_configs

# --- 5. Main Execution ---
def main():
    try:
        pso_params, selected_configs = get_user_input()
    except:
        pso_params, selected_configs = GLOBAL_PARAMS, ALL_FUNCTION_CONFIGS

    results_table = []
    plt.figure(figsize=(12, 7))

    for config in selected_configs:
        final_fitnesses = []
        all_run_histories = []
        
        print(f"Analyzing: {config['name']}...", end="\r")
        for run in range(pso_params['N_RUNS']):
            f_gval, run_history = run_pso_once(config["func"], config["min_range"], config["max_range"], 12345+run, pso_params, config.get("params", {}))
            final_fitnesses.append(f_gval)
            all_run_histories.append(run_history)
        
        fitnesses = np.array(final_fitnesses)
        results_table.append([config["name"], f"{np.min(fitnesses):.6e}", f"{np.mean(fitnesses):.6e}", f"{np.median(fitnesses):.6e}", f"{np.std(fitnesses):.6e}"])
        
        avg_history = np.mean(all_run_histories, axis=0)
        plt.semilogy(avg_history, label=config["name"])

    print("\n" + "="*80)
    print("PSO PERFORMANCE STATISTICAL SUMMARY (CEC FUNCTIONS)")
    print(tabulate(results_table, headers=["Function Name", "Minimum", "Mean", "Median", "Std Dev"], tablefmt="fancy_grid"))
    
    plt.title(f"PSO Convergence Curves (D={pso_params['D']}, N={pso_params['N']})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost (Log Scale)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()