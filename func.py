import math

def sphere(x):
    return sum((i ** 2 for i in x))

def csendes(x):
    """
    Csendes Function - Simple smooth function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return sum(xi**6 * (2 + math.sin(1 / (xi + 1e-10))) for xi in x)

def sum_of_different_powers(x):
    return sum((abs(xi) ** (i + 2) for i, xi in enumerate(x)))

def step(x):
    return sum((int(i) ** 2 for i in x))

def powell(x):
    """
    Powell Function - Non-separable
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-4, 5]
    Note: Works best with dims divisible by 4
    """
    n = len(x)
    result = 0
    for i in range(0, n - 3, 4):
        result += (x[i] + 10*x[i+1])**2
        result += 5 * (x[i+2] - x[i+3])**2
        result += (x[i+1] - 2*x[i+2])**4
        result += 10 * (x[i] - x[i+3])**4
    return result

def dixon_price(x):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum(((i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, n)))
    return term1 + term2

def schumer_steiglitz(x):
    return sum((xi ** 4 for xi in x))

def sixth_power(x):
    return sum((xi ** 6 for xi in x))

def quartic(x):
    return sum(((i + 1) * xi ** 4 for i, xi in enumerate(x)))

def exponential(x):
    """
    Exponential Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return -math.exp(-0.5 * sum(xi**2 for xi in x)) + 1

def rastrigin(x):
    n = len(x)
    return 10 * n + sum((xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x))

def ackley(x):
    n = len(x)
    sum1 = sum((xi ** 2 for xi in x))
    sum2 = sum((math.cos(2 * math.pi * xi) for xi in x))
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e

def griewank(x):
    sum_term = sum((xi ** 2 for xi in x)) / 4000
    prod_term = 1
    for i, xi in enumerate(x):
        prod_term *= math.cos(xi / math.sqrt(i + 1))
    return sum_term - prod_term + 1

def schwefel(x):
    n = len(x)
    return 418.9829 * n - sum((xi * math.sin(math.sqrt(abs(xi))) for xi in x))

def levy(x):
    n = len(x)
    w = [1 + (xi - 1) / 4 for xi in x]
    term1 = math.sin(math.pi * w[0]) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)
    term2 = sum(((w[i] - 1) ** 2 * (1 + 10 * math.sin(math.pi * w[i] + 1) ** 2) for i in range(n - 1)))
    return term1 + term2 + term3

def styblinski_tang(x):
    return sum((xi ** 4 - 16 * xi ** 2 + 5 * xi for xi in x)) / 2

def alpine(x):
    return sum((abs(xi * math.sin(xi) + 0.1 * xi) for xi in x))

def schaffer_f6(x):
    result = 0
    for i in range(len(x) - 1):
        ss = x[i] ** 2 + x[i + 1] ** 2
        result += 0.5 + (math.sin(math.sqrt(ss)) ** 2 - 0.5) / (1 + 0.001 * ss) ** 2
    return result

def rosenbrock(x):
    n = len(x)
    return sum((100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(n - 1)))

def drop_wave(x):
    sum_sq = sum((xi ** 2 for xi in x))
    numerator = 1 + math.cos(12 * math.sqrt(sum_sq))
    denominator = 0.5 * sum_sq + 2
    return -numerator / denominator