"""
Benchmark Functions for Algorithm Comparison

Contains:
- 10 Unimodal Functions (smooth, single global minimum)
- 10 Multimodal Functions (multiple local minima, good for testing all 4 algorithms)
"""

import math

# ==============================================================================
# UNIMODAL FUNCTIONS (10)
# ==============================================================================

def sphere(x):
    """
    Sphere Function - Simple unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5.12, 5.12]
    """
    return sum(i**2 for i in x)


def sum_of_squares(x):
    """
    Sum of Squares Function - Unimodal, easy baseline
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum((i + 1) * xi**2 for i, xi in enumerate(x))


def sum_of_different_powers(x):
    """
    Sum of Different Powers Function - Very easy
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return sum(abs(xi) ** (i + 2) for i, xi in enumerate(x))


def step(x):
    """
    Step Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    return sum(int(i)**2 for i in x)


def zakharov(x):
    """
    Zakharov Function - Unimodal, bowl-shaped
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5, 10]
    """
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(0.5 * (i + 1) * xi for i, xi in enumerate(x))
    return sum1 + sum2**2 + sum2**4


def dixon_price(x):
    """
    Dixon–Price Function
    Global minimum: f(x*) = 0
    Search domain: [-10, 10]
    """
    n = len(x)
    term1 = (x[0] - 1)**2
    term2 = sum((i + 1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, n))
    return term1 + term2


def schumer_steiglitz(x):
    """
    Schumer Steiglitz Function - Sum of fourth powers
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum(xi**4 for xi in x)


def sixth_power(x):
    """
    Sixth Power Function - Smooth unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return sum(xi**6 for xi in x)


def quartic(x):
    """
    Quartic Function (De Jong's F4)
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1.28, 1.28]
    """
    return sum((i + 1) * xi**4 for i, xi in enumerate(x))


def rotated_hyper_ellipsoid(x):
    """
    Rotated Hyper-Ellipsoid Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-65.536, 65.536]
    """
    n = len(x)
    result = 0
    for i in range(n):
        result += sum(x[j]**2 for j in range(i + 1))
    return result


# ==============================================================================
# MULTIMODAL FUNCTIONS (10)
# These are well-balanced functions suitable for PSO, GA, Tabu, and SA
# ==============================================================================

def rastrigin(x):
    """
    Rastrigin Function - Classic multimodal benchmark
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5.12, 5.12]
    Many regularly distributed local minima, great for testing exploration
    """
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def ackley(x):
    """
    Ackley Function - Multimodal with nearly flat outer region
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-32.768, 32.768]
    Tests algorithm's ability to escape flat regions
    """
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e


def griewank(x):
    """
    Griewank Function - Widespread local minima
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-600, 600]
    Product term creates interaction between variables
    """
    sum_term = sum(xi**2 for xi in x) / 4000
    prod_term = 1
    for i, xi in enumerate(x):
        prod_term *= math.cos(xi / math.sqrt(i + 1))
    return sum_term - prod_term + 1


def schwefel(x):
    """
    Schwefel Function - Deceptive with distant global minimum
    Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    Search domain: [-500, 500]
    Global minimum far from local minima, tests exploration
    """
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)


def levy(x):
    """
    Levy Function - Smooth multimodal
    Global minimum: f(1, ..., 1) = 0
    Search domain: [-10, 10]
    Moderate difficulty, good for all algorithms
    """
    n = len(x)
    w = [(1 + (xi - 1) / 4) for xi in x]
    
    term1 = math.sin(math.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + math.sin(2 * math.pi * w[-1])**2)
    
    term2 = sum((w[i] - 1)**2 * (1 + 10 * math.sin(math.pi * w[i] + 1)**2) 
                for i in range(n - 1))
    
    return term1 + term2 + term3


def styblinski_tang(x):
    """
    Styblinski-Tang Function - Simple multimodal
    Global minimum: f(-2.903534, ..., -2.903534) = -39.16617 * d
    Search domain: [-5, 5]
    Relatively easy multimodal, all algorithms perform well
    """
    return sum(xi**4 - 16*xi**2 + 5*xi for xi in x) / 2


def alpine(x):
    """
    Alpine Function - Moderate multimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    Non-separable, tests variable interaction
    """
    return sum(abs(xi * math.sin(xi) + 0.1 * xi) for xi in x)


def schaffer_f6(x):
    """
    Schaffer F6 Function (generalized)
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    Challenging with circular ridges, good for testing fine-tuning
    """
    result = 0
    for i in range(len(x) - 1):
        ss = x[i]**2 + x[i+1]**2
        result += 0.5 + (math.sin(math.sqrt(ss))**2 - 0.5) / (1 + 0.001 * ss)**2
    return result


def rosenbrock(x):
    """
    Rosenbrock Function - Classic banana-shaped valley
    Global minimum: f(1, ..., 1) = 0
    Search domain: [-5, 10]
    Tests convergence in narrow curved valley
    """
    n = len(x)
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(n - 1))


def drop_wave(x):
    """
    Drop-Wave Function (generalized)
    Global minimum: f(0, ..., 0) = -1
    Search domain: [-5.12, 5.12]
    Concentric waves pattern, tests global search
    """
    sum_sq = sum(xi**2 for xi in x)
    numerator = 1 + math.cos(12 * math.sqrt(sum_sq))
    denominator = 0.5 * sum_sq + 2
    return -numerator / denominator
