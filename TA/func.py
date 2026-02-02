import math

def sphere(x):
    return sum((i ** 2 for i in x))

def sum_of_squares(x):
    return sum(((i + 1) * xi ** 2 for i, xi in enumerate(x)))

def sum_of_different_powers(x):
    return sum((abs(xi) ** (i + 2) for i, xi in enumerate(x)))

def step(x):
    return sum((int(i) ** 2 for i in x))

def brown(x):
    n = len(x)
    return sum(((x[i] ** 2) ** (x[i + 1] ** 2 + 1) + (x[i + 1] ** 2) ** (x[i] ** 2 + 1) for i in range(n - 1)))

def zakharov(x):
    sum1 = sum((xi ** 2 for xi in x))
    sum2 = sum((0.5 * (i + 1) * xi for i, xi in enumerate(x)))
    return sum1 + sum2 ** 2 + sum2 ** 4

def dixon_price(x):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum(((i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, n)))
    return term1 + term2

def schumer_steiglitz(x):
    return sum((xi ** 4 for xi in x))

def csendes(x):
    return sum((xi ** 6 * (2 + math.sin(1 / (xi + 1e-10))) for xi in x))

def sixth_power(x):
    return sum((xi ** 6 for xi in x))

def powell(x):
    n = len(x)
    result = 0
    for i in range(0, n - 3, 4):
        result += (x[i] + 10 * x[i + 1]) ** 2
        result += 5 * (x[i + 2] - x[i + 3]) ** 2
        result += (x[i + 1] - 2 * x[i + 2]) ** 4
        result += 10 * (x[i] - x[i + 3]) ** 4
    return result

def quartic(x):
    return sum(((i + 1) * xi ** 4 for i, xi in enumerate(x)))

def rotated_hyper_ellipsoid(x):
    n = len(x)
    result = 0
    for i in range(n):
        result += sum((x[j] ** 2 for j in range(i + 1)))
    return result

def discus(x):
    return 1000000.0 * x[0] ** 2 + sum((xi ** 2 for xi in x[1:]))

def exponential(x):
    return -math.exp(-0.5 * sum((xi ** 2 for xi in x))) + 1