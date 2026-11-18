"""
markov_tools.py
Инструменты для анализа марковских цепей методом логарифмической нормы.
Основано на: Зейфман А.И., Сатин Я.А., Королев В.Ю.
"Марковские цепи с непрерывным временем: асимптотические свойства
и их применение в теории массового обслуживания"
"""

import numpy as np

# Порог для проверки существенной неотрицательности
NONNEG_TOL = 1e-10


def build_D(d):
    """
    Построить верхне-треугольную матрицу D из формулы (1.26).

    Параметры
    ---------
    d : array-like, длина n
        Положительные веса d_1, ..., d_n.

    Возвращает
    ----------
    D : ndarray, форма (n, n)
        Верхне-треугольная матрица: D[i, j] = d[i] при j >= i, иначе 0.
    """
    d = np.asarray(d, dtype=float)
    if np.any(d <= 0):
        raise ValueError("Все d_i должны быть положительными")
    n = len(d)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i:] = d[i]
    return D


def build_D_inv(d):
    """
    Построить D^{-1} для верхне-треугольной D из (1.26).

    Явная формула: D^{-1}[i, i] = 1/d_i,
                   D^{-1}[i, i+1] = -1/d_{i+1},
                   остальные элементы = 0.

    Параметры
    ---------
    d : array-like, длина n

    Возвращает
    ----------
    D_inv : ndarray, форма (n, n)
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    D_inv = np.zeros((n, n))
    for i in range(n):
        D_inv[i, i] = 1.0 / d[i]
        if i + 1 < n:
            D_inv[i, i + 1] = -1.0 / d[i + 1]
    return D_inv


def compute_B(A):
    """
    Вычислить B из A по формуле (1.21): b_ij = a_ij - a_i0.

    Параметры
    ---------
    A : ndarray, форма (n+1, n+1)
        Транспонированная инфинитезимальная матрица
        (состояния 0, 1, ..., n).

    Возвращает
    ----------
    B : ndarray, форма (n, n)
        Матрица после исключения нулевого состояния
        (состояния 1, ..., n).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0] - 1
    B = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            B[i - 1, j - 1] = A[i, j] - A[i, 0]
    return B


def compute_B_fast(A):
    A = np.asarray(A, dtype=float)
    A_reduced = A[1:, 1:]          # угол (состояния 1..n)
    a_i0 = A[1:, 0:1]              # столбец A[i, 0] для i >= 1
    return A_reduced - a_i0        # вещание по строкам


def compute_B_star(B, d):
    """
    Вычислить B** = D B D^{-1}.

    Параметры
    ---------
    B : ndarray, форма (n, n)
    d : array-like, длина n

    Возвращает
    ----------
    B_star : ndarray, форма (n, n)
    D : ndarray, форма (n, n)
    D_inv : ndarray, форма (n, n)
    """
    D = build_D(d)
    D_inv = build_D_inv(d)
    B_star = D @ B @ D_inv
    return B_star, D, D_inv


def log_norm(B):
    """
    Логарифмическая норма в l_1 (формула (1.12)).

    gamma(B) = sup_j ( b_jj + sum_{i != j} |b_ij| )
    """
    B = np.asarray(B, dtype=float)
    n = B.shape[0]
    gamma = -np.inf
    for j in range(n):
        col = B[:, j].copy()
        col[j] = 0  # убираем диагональ из суммы модулей
        s = B[j, j] + np.sum(np.abs(col))
        gamma = max(gamma, s)
    return gamma


def is_essentially_nonneg(B, tol=NONNEG_TOL):
    """
    Проверить, что B существенно неотрицательна
    (все внедиагональные элементы >= -tol).
    """
    B = np.asarray(B, dtype=float)
    n = B.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and B[i, j] < -tol:
                return False
    return True


def is_essentially_nonneg_fast(B, tol=NONNEG_TOL):
    B = np.asarray(B, dtype=float)
    mask = ~np.eye(B.shape[0], dtype=bool)
    return np.all(B[mask] >= -tol)


def analyze(A, d, verbose=True):
    """
    Полный конвейер: A -> B -> D, D^{-1} -> B** -> gamma -> beta.

    Возвращает словарь со всеми матрицами и результатами.
    """
    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float)

    B = compute_B(A)
    B_star, D, D_inv = compute_B_star(B, d)
    gamma = log_norm(B_star)
    beta = -gamma

    result = {
        'A': A,
        'B': B,
        'D': D,
        'D_inv': D_inv,
        'B_star': B_star,
        'gamma': gamma,
        'beta': beta,
        'd': d,
        'ess_nonneg': is_essentially_nonneg(B_star),
    }

    if verbose:
        print(f"  gamma(B**) = {gamma:.6f}")
        print(f"  beta = -gamma = {beta:.6f}")
        print(f"  B** существенно неотрицательна: {result['ess_nonneg']}")

    return result


def print_matrix(M, name="M", fmt="{:9.4f}"):
    M = np.asarray(M, dtype=float)
    n, m = M.shape
    print(f"\n{name} ({n}×{m}):")
    for row in M:
        print("  [" + ", ".join(fmt.format(x) for x in row) + "]")
