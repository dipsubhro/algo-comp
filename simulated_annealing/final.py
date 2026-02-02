import numpy as np

base_seed = 84748

for run in range(25):
    np.random.seed(base_seed + run)
 


import math

def sphere(x):
    return sum(i**2 for i in x)

def sum_of_squares(x):
    return sum((i + 1) * xi**2 for i, xi in enumerate(x))

def sum_of_different_powers(x):
    return sum(abs(xi) ** (i + 2) for i, xi in enumerate(x))

def step(x):
    return sum(int(i)**2 for i in x)

def brown(x):
    n = len(x)
    return sum((x[i]**2) ** (x[i+1]**2 + 1) + (x[i+1]**2) ** (x[i]**2 + 1) for i in range(n - 1))

def zakharov(x):
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(0.5 * (i + 1) * xi for i, xi in enumerate(x))
    return sum1 + sum2**2 + sum2**4

def dixon_price(x):
    n = len(x)
    term1 = (x[0] - 1)**2
    term2 = sum((i + 1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, n))
    return term1 + term2

def schumer_steiglitz(x):
    return sum(xi**4 for xi in x)

def csendes(x):
    return sum(xi**6 * (2 + math.sin(1 / (xi + 1e-10))) for xi in x)

def sixth_power(x):
    return sum(xi**6 for xi in x)

def powell(x):
    n = len(x)
    result = 0
    for i in range(0, n - 3, 4):
        result += (x[i] + 10*x[i+1])**2
        result += 5 * (x[i+2] - x[i+3])**2
        result += (x[i+1] - 2*x[i+2])**4
        result += 10 * (x[i] - x[i+3])**4
    return result

def quartic(x):
    return sum((i + 1) * xi**4 for i, xi in enumerate(x))

def rotated_hyper_ellipsoid(x):
    n = len(x)
    result = 0
    for i in range(n):
        result += sum(x[j]**2 for j in range(i + 1))
    return result

def discus(x):
    return 1e6 * x[0]**2 + sum(xi**2 for xi in x[1:])

def exponential(x):
    return -math.exp(-0.5 * sum(xi**2 for xi in x)) + 1



# -------------------------------------------------
# Function Map & Bounds
# -------------------------------------------------

functions = {
    "sphere": sphere,
    "sum_of_squares": sum_of_squares,
    "sum_of_different_powers": sum_of_different_powers,
    "step": step,
    "brown": brown,
    "zakharov": zakharov,
    "dixon_price": dixon_price,
    "schumer_steiglitz": schumer_steiglitz,
    "csendes": csendes,
    "sixth_power": sixth_power,
    "powell": powell,
    "quartic": quartic,
    "rotated_hyper_ellipsoid": rotated_hyper_ellipsoid,
    "discus": discus,
    "exponential": exponential
}





bounds = {
    "sphere": (-5.12, 5.12),
    "sum_of_squares": (-10, 10),
    "sum_of_different_powers": (-1, 1),
    "step": (-100, 100),
    "brown": (-1, 4),
    "zakharov": (-5, 10),
    "dixon_price": (-10, 10),
    "schumer_steiglitz": (-100, 100),
    "csendes": (-1, 1),
    "sixth_power": (-1, 1),
    "powell": (-4, 5),
    "quartic": (-1.28, 1.28),
    "rotated_hyper_ellipsoid": (-65.536, 65.536),
    "discus": (-100, 100),
    "exponential": (-1, 1)
}




def simulated_annealing(cost_func, dim=5, max_iter=1500):

    ip, r, fp = 0.9, 0.95, 0.01
    L = 50
    T = 1000.0
    T_min = 1e-6   #  FREEZING TEMPERATURE


    S = np.random.uniform(-5, 5, dim)
    S_star = S.copy()

    for _ in range(max_iter):

        j = 0
        for _ in range(L):
            S_prime = S + np.random.uniform(-0.5, 0.5, dim)
            delta = cost_func(S_prime) - cost_func(S)

            if delta <= 0 or np.random.rand() < np.exp(-delta / T):
                S = S_prime
                j += 1
                if cost_func(S) < cost_func(S_star):
                    S_star = S.copy()

        acc_ratio = j / L
        T = T/2 if acc_ratio > ip else r*T
        # 🧊 FREEZING CRITERION (HERE!)
        if T < T_min:
            break

    return cost_func(S_star)




runs = 25

print(f"{'Function':<15} {'Min':<12} {'Mean':<12} {'Median':<12} {'max':<12} {'Std Dev':<12}")
print("-"*65)

for name, func in functions.items():
    results = [simulated_annealing(func) for _ in range(runs)]

    print(f"{name:<15} "
          f"{np.min(results):<12.6f} "
          f"{np.mean(results):<12.6f} "
          f"{np.median(results):<12.6f} "
          f"{np.max(results):<12.6f} "
          f"{np.std(results):<12.6f}")
