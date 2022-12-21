import collections
import scipy.optimize as opt
from scipy.optimize import fsolve
from math import *
from prettytable import PrettyTable


def Hi(v):
    """
    inverse of binary entropy function
    """
    if v == 1:
        return 0.5
    if v < 0.00001:
        return 0
    return fsolve(lambda x: v - (-x * log2(x) - (1 - x) * log2(1 - x)), 0.0000001)[0]


def H(c):
    """
    binary entropy function
    """
    if c == 0.0 or c == 1.0:
        return 0.0

    if c < 0.0 or c > 1.0:
        return -1000

    return -(c * log2(c) + (1 - c) * log2(1 - c))


set_vars = collections.namedtuple("TEST", "l p")


def wrap(f, g):
    def inner(x):
        return f(g(*x))

    return inner


def re(f):
    return wrap(f, set_vars)


def FindOptValues_for_a_1(w, k, iters, start_values = None):
    '''
    Для высоты дерева a = 1
    '''
    prob = lambda x: (1 - k - x.l) * H((w - x.p) / (1 - k - x.l)) + w - x.p - log2(3) * (1 - k - x.l)
    Lsize = lambda x: 0.5 * (k + x.l) * H(0.5 * x.p / (k + x.l)) + 0.5 * (k + x.l - 0.5 * x.p) * H(
        x.p / (2 * k + 2 * x.l - x.p))
    Lout = lambda x: 2 * Lsize(x) - x.l * log2(3)
    TSS = lambda x: max(Lsize(x), Lout(x))
    T = lambda x: TSS(x) + max(0, -prob(x) - Lout(x))

    def time(x):
        x = set_vars(*x)
        return T(x)

    constraints_re = [
        # ограничения на переменные
        {"type": "ineq", "fun": re(lambda x: 1 - k - x.l)},  # n >= k + l
        {"type": "ineq", "fun": re(lambda x: w - x.p)},  # w >= p
        {"type": "ineq", "fun": re(lambda x: 1 - k - x.l - w + x.p)},  # n - k - l >= w - p
        {"type": "ineq", "fun": re(lambda x: k + x.l - x.p)},  # k + l >= p

        # ограничения на функции
        {"type": "ineq", "fun": re(lambda x: -prob(x))},  # 0 <= 2^prob(x) <= 1
        {"type": "ineq", "fun": re(lambda x: Lout(x))},  # Lout >= 0
        # {"type": "eq", "fun": re(lambda x: Lsize(x) - Lout(x))},
    ]

    if start_values:
        result = opt.minimize(time,
                              [start_values[0]+0.0001, start_values[1]+0.0001],
                              bounds=[(0.00000, 0.7), (0.01, w - 0.05)], tol=1e-10,
                              constraints=constraints_re, options={'maxiter': 1000})

    else:
        result = opt.minimize(time,
                          [0.1, k + 0.01],
                          bounds=[(0.00000, 0.7), (0.01, w-0.05)], tol=1e-10,
                          constraints=constraints_re, options={'maxiter': 1000})
    min_time = result.fun
    fin_result = result

    for i in range(0, iters):
        result = opt.minimize(time,
                              [result.x[0] + 0.001, result.x[1] + 0.0001],
                              bounds=[(0.00000, 0.7), (0.01, w-0.05)], tol=1e-10,
                              constraints=constraints_re, options={'maxiter': 1000})

        if result.fun < min_time and result.success:
            min_time = result.fun
            fin_result = result

    y = set_vars(*fin_result.x)
    return [fin_result.fun, fin_result.x[0], fin_result.x[1], prob(y), Lsize(y), 0, Lout(y), TSS(y)]


def FindOptValues_for_a_2(w, k, iters, start_values = None):
    '''
    Для высоты дерева a = 2
    '''
    prob = lambda x: (1 - k - x.l) * H((w - x.p) / (1 - k - x.l)) + w - x.p - log2(3) * (1 - k - x.l)
    Lsize = lambda x: 0.25 * (k + x.l) * H(0.5 * x.p / (k + x.l)) + 0.25 * (k + x.l - 0.5 * x.p) * H(
        x.p / (2 * k + 2 * x.l - x.p))
    Linterm = lambda x: 2 * Lsize(x) - x.l / 2 * log2(3)
    Lout = lambda x: 2 * Linterm(x) - x.l / 2 * log2(3)
    TSS = lambda x: max(Lsize(x), Linterm(x), Lout(x))
    T = lambda x: TSS(x) + max(0, -prob(x) - Lout(x))

    def time(x):
        x = set_vars(*x)
        return T(x)

    constraints_re = [
        # ограничения на переменные
        {"type": "ineq", "fun": re(lambda x: 1 - k - x.l)},  # n >= k + l
        {"type": "ineq", "fun": re(lambda x: w - x.p)},  # w >= p
        {"type": "ineq", "fun": re(lambda x: 1 - k - x.l - w + x.p)},  # n - k - l >= w - p
        {"type": "eq", "fun": re(lambda x: k + x.l - x.p)},  # k + l >= p

        # ограничения на функции
        {"type": "ineq", "fun": re(lambda x: -prob(x))},  # 0 <= 2^prob(x) <= 1
        {"type": "ineq", "fun": re(lambda x: Lout(x))},  # Lout >= 0
        # {"type": "eq", "fun": re(lambda x: Lsize(x) - Lout(x))},
        #{"type": "ineq", "fun": re(lambda x: (k+x.l)*H(x.p/(k+x.l))+x.p-x.l*log2(3) - Lout(x))}
    ]
    if start_values:
        result = opt.minimize(time,
                              [start_values[0] + 0.0001, start_values[1] + 0.0001],
                              bounds=[(0.00000, 0.7), (0.01, w - 0.05)], tol=1e-10,
                              constraints=constraints_re, options={'maxiter': 1000})
    else:
        result = opt.minimize(time,
                              [0.1, k + 0.01],
                              bounds=[(0.00000, 0.7), (0.01, w-0.05)], tol=1e-10,
                              constraints=constraints_re, options={'maxiter': 500})

    min_time = result.fun
    fin_result = result

    for i in range(0, iters):
        result = opt.minimize(time,
                              [result.x[0] + 0.001, result.x[1] + 0.0001],
                              bounds=[(0.00000, 0.7),(0.01, w-0.05)], tol=1e-10,
                              constraints=constraints_re, options={'maxiter': 500})

        if result.fun < min_time and result.success:
            min_time = result.fun
            fin_result = result

    y = set_vars(*fin_result.x)
    return [fin_result.fun,fin_result.x[0],fin_result.x[1],prob(y),Lsize(y),Linterm(y),Lout(y),TSS(y)]


def FindOptValues_for_a_3(w, k, iters):
    '''
    Для высоты дерева a = 3
    '''
    prob = lambda x: (1 - k - x.l) * H((w - x.p) / (1 - k - x.l)) + w - x.p - log2(3) * (1 - k - x.l)

    Lsize = lambda x: 0.125 * (k + x.l) * H(0.5 * x.p / (k + x.l)) + 0.125 * (k + x.l - 0.5 * x.p) * H(
        x.p / (2 * k + 2 * x.l - x.p))
    lambd = lambda x: x.l * log2(3) - 2 * Lsize(x)
    l0 = lambda x: (2*Lsize(x)-lambd(x))/log2(3)
    Linterm1 = lambda x: 2 * Lsize(x) - l0(x) * log2(3)
    Linterm2 = lambda x: 2 * Linterm1(x) - lambd(x)
    Lout = lambda x: 2 * Linterm2(x) - lambd(x)
    TSS = lambda x: max(Lsize(x), Linterm1(x), Linterm2(x), Lout(x))
    T = lambda x: TSS(x) + max(0, -prob(x) - Lout(x))

    def time(x):
        x = set_vars(*x)
        return T(x)

    constraints_re = [
        # ограничения на переменные
        {"type": "ineq", "fun": re(lambda x: 1 - k - x.l)},  # n >= k + l
        {"type": "ineq", "fun": re(lambda x: w - x.p)},  # w >= p
        {"type": "ineq", "fun": re(lambda x: 1 - k - x.l - w + x.p)},  # n - k - l >= w - p
        {"type": "eq",   "fun": re(lambda x: k + x.l - x.p)},  # k + l >= p

        # ограничения на функции
        {"type": "ineq", "fun": re(lambda x: -prob(x))},  # 0 <= 2^prob(x) <= 1
        {"type": "ineq", "fun": re(lambda x: Lout(x))},  # Lout >= 0
        {"type": "eq", "fun": re(lambda x: Lsize(x) - Lout(x))},
    ]

    result = opt.minimize(time,
                          [0.1, k + 0.01],
                          bounds=[(0.00000, 0.7), (0.01, w-0.05)], tol=1e-10,
                          constraints=constraints_re, options={'maxiter': 1000})

    min_time = result.fun
    fin_result = result

    for i in range(0, iters):
        result = opt.minimize(time,
                              [result.x[0] + 0.001, result.x[1] + 0.0001],
                              bounds=[(0.00000, 0.7),(0.01, w-0.05)], tol=1e-10,
                              constraints=constraints_re, options={'maxiter': 1000})

        if result.fun < min_time and result.success:
            min_time = result.fun
            fin_result = result

    y = set_vars(*fin_result.x)
    return [fin_result.fun,fin_result.x[0],fin_result.x[1],prob(y),Lsize(y),Linterm1(y),Lout(y),TSS(y)]


if __name__ == "__main__":
    k = 0.5  # relative code dimension
    # w = 1      # weight of solution
    iters = 100
    tmp_result_a_1 = []
    tmp_result_a_2 = []
    for w in [i / 100 for i in range(95, 96, 5)]:
        result_a_1 = FindOptValues_for_a_1(w, k, iters)
        result_a_2 = FindOptValues_for_a_2(w, k, iters)
        #result_a_3 = FindOptValues_for_a_3(w, k, iters)

        table = PrettyTable()
        table.add_column(f"w = {w}", ["time", "l", "p", "prob", "Lsize", "Linterm", "Lout", "TSS"])
        table.add_column("a = 1", result_a_1)
        table.add_column("a = 2", result_a_2)
        #table.add_column("a = 3", result_a_3)

        if w == 0.95:
            tmp_result_a_1 = result_a_1
            tmp_result_a_2 = result_a_2

            tmp_result_a_1 = [float(int(tmp_result_a_1[1] * 10 ** 6)) / 10 ** 6,
                              float(int(tmp_result_a_1[2] * 10 ** 6)) / 10 ** 6]
            tmp_result_a_2 = [float(int(tmp_result_a_2[1] * 10 ** 8)) / 10 ** 8,
                              float(int(tmp_result_a_2[2] * 10 ** 8)) / 10 ** 8]
        print(table)
        print()


    if tmp_result_a_1 !=[] and tmp_result_a_2 != []:
        result_a_1 = FindOptValues_for_a_1(1, k, iters,tmp_result_a_1)
        result_a_2 = FindOptValues_for_a_2(1, k, iters,tmp_result_a_2)
        result_a_3 = FindOptValues_for_a_3(1, k, iters)

        table = PrettyTable()
        table.add_column(f"w = 1", ["time", "l", "p", "prob", "Lsize", "Linterm", "Lout", "TSS"])
        table.add_column("a = 1", result_a_1)
        table.add_column("a = 2", result_a_2)
        table.add_column("a = 3", result_a_3)
        print(table)
        print()

# fun: 0.2696373180591388
