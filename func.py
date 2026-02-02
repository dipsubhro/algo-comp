import math

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
    Search domain: [-5.12, 5.12]
    """
    return sum(int(i)**2 for i in x)

def brown(x):
    """
    Brown Function - Smooth unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 4]
    """
    n = len(x)
    return sum((x[i]**2) ** (x[i+1]**2 + 1) + (x[i+1]**2) ** (x[i]**2 + 1) for i in range(n - 1))

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

def csendes(x):
    """
    Csendes Function - Simple smooth function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return sum(xi**6 * (2 + math.sin(1 / (xi + 1e-10))) for xi in x)

def sixth_power(x):
    """
    Sixth Power Function - Smooth, similar to Csendes
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return sum(xi**6 for xi in x)

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

def discus(x):
    """
    Discus (Tablet) Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    return 1e6 * x[0]**2 + sum(xi**2 for xi in x[1:])

def exponential(x):
    """
    Exponential Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return -math.exp(-0.5 * sum(xi**2 for xi in x)) + 1


# ==============================================================================
# ADDITIONAL UNIMODAL FUNCTIONS (10)
# ==============================================================================

def booth(x):
    """
    Booth Function - 2D Unimodal
    Global minimum: f(1, 3) = 0
    Search domain: [-10, 10]
    """
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def matyas(x):
    """
    Matyas Function - 2D Unimodal
    Global minimum: f(0, 0) = 0
    Search domain: [-10, 10]
    """
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]


def three_hump_camel(x):
    """
    Three-Hump Camel Function - 2D Unimodal
    Global minimum: f(0, 0) = 0
    Search domain: [-5, 5]
    """
    return 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2


def beale(x):
    """
    Beale Function - 2D Unimodal
    Global minimum: f(3, 0.5) = 0
    Search domain: [-4.5, 4.5]
    """
    return ((1.5 - x[0] + x[0]*x[1])**2 + 
            (2.25 - x[0] + x[0]*x[1]**2)**2 + 
            (2.625 - x[0] + x[0]*x[1]**3)**2)


def trid(x):
    """
    Trid Function - Unimodal
    Global minimum: f(x*) = -d(d+4)(d-1)/6 for d dimensions
    Search domain: [-d^2, d^2]
    """
    n = len(x)
    term1 = sum((xi - 1)**2 for xi in x)
    term2 = sum(x[i] * x[i-1] for i in range(1, n))
    return term1 - term2


def perm_d_beta(x, beta=10):
    """
    Perm d, beta Function - Unimodal
    Global minimum: f(1, 2, ..., d) = 0
    Search domain: [-d, d]
    """
    n = len(x)
    result = 0
    for i in range(1, n + 1):
        inner_sum = sum((j**i + beta) * ((x[j-1] / j)**i - 1) for j in range(1, n + 1))
        result += inner_sum**2
    return result


def schwefel_2_22(x):
    """
    Schwefel 2.22 Function - Unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    abs_sum = sum(abs(xi) for xi in x)
    abs_prod = 1
    for xi in x:
        abs_prod *= abs(xi)
    return abs_sum + abs_prod


def schwefel_2_21(x):
    """
    Schwefel 2.21 Function - Unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    return max(abs(xi) for xi in x)


def cigar(x):
    """
    Cigar Function - Unimodal, High condition number
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    return x[0]**2 + 1e6 * sum(xi**2 for xi in x[1:])


def sum_of_abs(x):
    """
    Sum of Absolute Values Function - Unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum(abs(xi) for xi in x)


# ==============================================================================
# MULTIMODAL FUNCTIONS (10)
# ==============================================================================

def rastrigin(x):
    """
    Rastrigin Function - Highly multimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5.12, 5.12]
    """
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def ackley(x):
    """
    Ackley Function - Multimodal with many local minima
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-32.768, 32.768]
    """
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e


def griewank(x):
    """
    Griewank Function - Multimodal with product term
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-600, 600]
    """
    sum_term = sum(xi**2 for xi in x) / 4000
    prod_term = 1
    for i, xi in enumerate(x):
        prod_term *= math.cos(xi / math.sqrt(i + 1))
    return sum_term - prod_term + 1


def schwefel(x):
    """
    Schwefel Function - Multimodal with distant global minimum
    Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    Search domain: [-500, 500]
    """
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)


def levy(x):
    """
    Levy Function - Multimodal
    Global minimum: f(1, ..., 1) = 0
    Search domain: [-10, 10]
    """
    n = len(x)
    w = [(1 + (xi - 1) / 4) for xi in x]
    
    term1 = math.sin(math.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + math.sin(2 * math.pi * w[-1])**2)
    
    term2 = sum((w[i] - 1)**2 * (1 + 10 * math.sin(math.pi * w[i] + 1)**2) 
                for i in range(n - 1))
    
    return term1 + term2 + term3


def michalewicz(x, m=10):
    """
    Michalewicz Function - Multimodal with steep valleys
    Global minimum depends on dimension (approx -1.8013 for d=2)
    Search domain: [0, π]
    """
    result = 0
    for i, xi in enumerate(x):
        result -= math.sin(xi) * (math.sin((i + 1) * xi**2 / math.pi))**(2 * m)
    return result


def eggholder(x):
    """
    Eggholder Function - 2D Multimodal with complex landscape
    Global minimum: f(512, 404.2319) ≈ -959.6407
    Search domain: [-512, 512]
    """
    result = 0
    for i in range(len(x) - 1):
        result -= (x[i+1] + 47) * math.sin(math.sqrt(abs(x[i+1] + x[i]/2 + 47)))
        result -= x[i] * math.sin(math.sqrt(abs(x[i] - (x[i+1] + 47))))
    return result


def schaffer_n2(x):
    """
    Schaffer N.2 Function - 2D Multimodal
    Global minimum: f(0, 0) = 0
    Search domain: [-100, 100]
    """
    numerator = math.sin(x[0]**2 - x[1]**2)**2 - 0.5
    denominator = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + numerator / denominator


def styblinski_tang(x):
    """
    Styblinski-Tang Function - Multimodal
    Global minimum: f(-2.903534, ..., -2.903534) = -39.16617 * d
    Search domain: [-5, 5]
    """
    return sum(xi**4 - 16*xi**2 + 5*xi for xi in x) / 2


def alpine(x):
    """
    Alpine Function - Multimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum(abs(xi * math.sin(xi) + 0.1 * xi) for xi in x)

