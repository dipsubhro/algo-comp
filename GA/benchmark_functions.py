import math


def sphere(x):
    """
    Sphere Function - Simple unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5.12, 5.12]
    """
    return sum(xi**2 for xi in x)


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
    # FIX: was sum(int(i)**2 for i in x) — wrong variable, wrong operation
    return sum(math.floor(abs(xi)) for xi in x)


def brown(x):
    """
    Brown Function - Smooth unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 4]
    """
    # FIX: was (x[i]**2) ** (x[i+1]**2 + 1) — exponentiation, not multiplication
    n = len(x)
    return sum((x[i]**2) * (x[i+1]**2 + 1) + (x[i+1]**2) * (x[i]**2 + 1) for i in range(n - 1))


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
    Dixon-Price Function
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


# ==============================================================
#   LOOKUP DICTIONARIES
# ==============================================================

benchmark_functions = {
    "Sphere":                   sphere,
    "Sum_of_Squares":           sum_of_squares,
    "Sum_of_Diff_Powers":       sum_of_different_powers,
    "Step":                     step,
    "Brown":                    brown,
    "Zakharov":                 zakharov,
    "Dixon_Price":              dixon_price,
    "Schumer_Steiglitz":        schumer_steiglitz,
    "Csendes":                  csendes,
    "Sixth_Power":              sixth_power,
    "Powell":                   powell,
    "Quartic":                  quartic,
    "Rotated_Hyper_Ellipsoid":  rotated_hyper_ellipsoid,
    "Discus":                   discus,
    "Exponential":              exponential,
}

function_bounds = {
    "Sphere":                   (-5.12,    5.12),
    "Sum_of_Squares":           (-10,      10),
    "Sum_of_Diff_Powers":       (-1,       1),
    "Step":                     (-5.12,    5.12),
    "Brown":                    (-1,       4),
    "Zakharov":                 (-5,       10),
    "Dixon_Price":              (-10,      10),
    "Schumer_Steiglitz":        (-10,      10),
    "Csendes":                  (-1,       1),
    "Sixth_Power":              (-1,       1),
    "Powell":                   (-4,       5),
    "Quartic":                  (-1.28,    1.28),
    "Rotated_Hyper_Ellipsoid":  (-65.536,  65.536),
    "Discus":                   (-100,     100),
    "Exponential":              (-1,       1),
}

function_properties = {
    "Sphere":                   {"type": "Unimodal",  "separable": True,  "min_at": "(0,...,0)", "min_val": 0},
    "Sum_of_Squares":           {"type": "Unimodal",  "separable": True,  "min_at": "(0,...,0)", "min_val": 0},
    "Sum_of_Diff_Powers":       {"type": "Unimodal",  "separable": True,  "min_at": "(0,...,0)", "min_val": 0},
    "Step":                     {"type": "Discontinuous", "separable": True,  "min_at": "[-1,1]^n", "min_val": 0},
    "Brown":                    {"type": "Unimodal",  "separable": False, "min_at": "(0,...,0)", "min_val": 0},
    "Zakharov":                 {"type": "Unimodal",  "separable": False, "min_at": "(0,...,0)", "min_val": 0},
    "Dixon_Price":              {"type": "Unimodal",  "separable": False, "min_at": "x_i=2^(-(2^i-2)/2^i)", "min_val": 0},
    "Schumer_Steiglitz":        {"type": "Unimodal",  "separable": True,  "min_at": "(0,...,0)", "min_val": 0},
    "Csendes":                  {"type": "Multimodal","separable": True,  "min_at": "(0,...,0)", "min_val": 0},
    "Sixth_Power":              {"type": "Unimodal",  "separable": True,  "min_at": "(0,...,0)", "min_val": 0},
    "Powell":                   {"type": "Unimodal",  "separable": False, "min_at": "(0,...,0)", "min_val": 0},
    "Quartic":                  {"type": "Unimodal",  "separable": True,  "min_at": "(0,...,0)", "min_val": 0},
    "Rotated_Hyper_Ellipsoid":  {"type": "Unimodal",  "separable": False, "min_at": "(0,...,0)", "min_val": 0},
    "Discus":                   {"type": "Unimodal",  "separable": True,  "min_at": "(0,...,0)", "min_val": 0},
    "Exponential":              {"type": "Unimodal",  "separable": False, "min_at": "(0,...,0)", "min_val": 0},
}


# ==============================================================
#   TEST
# ==============================================================
if __name__ == "__main__":
    zero_5  = [0.0] * 5
    zero_4  = [0.0] * 4   # Powell needs multiple of 4
    dp_min  = [2 ** (-(2**i - 2) / 2**i) for i in range(1, 6)]

    functions = [
        ("sphere",                  sphere,                  zero_5),
        ("sum_of_squares",          sum_of_squares,          zero_5),
        ("sum_of_different_powers", sum_of_different_powers, zero_5),
        ("step",                    step,                    zero_5),
        ("brown",                   brown,                   zero_5),
        ("zakharov",                zakharov,                zero_5),
        ("dixon_price",             dixon_price,             dp_min),
        ("schumer_steiglitz",       schumer_steiglitz,       zero_5),
        ("csendes",                 csendes,                 zero_5),
        ("sixth_power",             sixth_power,             zero_5),
        ("powell",                  powell,                  zero_4),
        ("quartic",                 quartic,                 zero_5),
        ("rotated_hyper_ellipsoid", rotated_hyper_ellipsoid, zero_5),
        ("discus",                  discus,                  zero_5),
        ("exponential",             exponential,             zero_5),
    ]

    print("=" * 55)
    print("  BENCHMARK FUNCTIONS — VERIFICATION (5D)")
    print("=" * 55)
    print(f"  {'Function':<28} {'Result':<12} {'Status'}")
    print("-" * 55)

    for name, func, pt in functions:
        val = func(pt)
        ok  = math.isclose(val, 0.0, abs_tol=1e-9)
        print(f"  {name:<28} {val:<12.6f} {'✓' if ok else '✗'}")

    print("-" * 55)