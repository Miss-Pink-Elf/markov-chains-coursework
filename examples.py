import numpy as np


def example_MM1(lam=1.0, mu=4.0, S=5):
    """
    M/M/1 с усечением до S состояний (0, 1, ..., S).

    Интенсивности: lambda_n = lam, mu_n = mu (n >= 1).
    Точное beta (бесконечный случай): (sqrt(mu) - sqrt(lam))^2.
    Точное beta (конечный случай): lam + mu - 2*sqrt(lam*mu)*cos(pi/(S+1)).
    """
    n = S + 1
    A = np.zeros((n, n))
    A[0, 0] = -lam
    A[0, 1] = mu
    for i in range(1, S):
        A[i, i - 1] = lam
        A[i, i] = -(lam + mu)
        A[i, i + 1] = mu
    A[S, S - 1] = lam
    A[S, S] = -mu

    delta = np.sqrt(mu / lam)
    d = [delta ** (i - 1) for i in range(1, S + 1)]

    return A, d


def example_MMNN(lam=1.0, mu=1.0, N=3):
    """
    M/M/N/N (система Эрланга с отказами), N приборов.
    Интенсивности: lambda_n = lam, mu_n = n*mu.

    Приближённый выбор d: delta = sqrt(N*mu/lam) (эвристика).
    Точный beta можно получить из теоремы 9 (рекурсия),
    см. функцию find_beta_MMNN_exact ниже.
    """
    n = N + 1
    A = np.zeros((n, n))
    A[0, 0] = -lam
    A[0, 1] = mu
    for i in range(1, N):
        A[i, i - 1] = lam
        A[i, i] = -(lam + i * mu)
        A[i, i + 1] = (i + 1) * mu
    A[N, N - 1] = lam
    A[N, N] = -N * mu

    delta = np.sqrt(N * mu / lam)
    d = [delta ** (i - 1) for i in range(1, N + 1)]

    return A, d


def find_beta_MMNN_exact(lam, mu, N):
    """
    Точное beta для M/M/N/N по теореме 9 (рекурсия (2.14)).
    Возвращает beta = mu + lam * (1 - delta_1*).
    """
    from scipy.optimize import brentq

    def delta_N(delta_1):
        if delta_1 <= 0 or delta_1 >= 1:
            return np.inf
        deltas = [delta_1]
        for k in range(1, N):
            d_next = delta_1 + (mu / lam) * k * (1 - 1 / deltas[-1])
            deltas.append(d_next)
        return deltas[-1]

    xs = np.linspace(0.5, 0.999, 500)
    for i in range(len(xs) - 1):
        f1 = delta_N(xs[i])
        f2 = delta_N(xs[i + 1])

        if np.isfinite(f1) and np.isfinite(f2) and f1*f2 < 0:
            delta_1_star = brentq(delta_N, xs[i], xs[i + 1])
            return mu + lam * (1 - delta_1_star)

    raise ValueError(f"Не удалось найти beta для M/M/{N}/{N}")


def example_class2(a=(1.0, 0.5), mu=1.0, S=5):
    """
    Класс II: групповые поступления (группы размеров 1, 2),
    одиночное обслуживание.

    a: интенсивности поступления групп размеров 1, 2.
    mu: интенсивность обслуживания.
    """
    a = np.asarray(a, dtype=float)
    n = S + 1

    A = np.zeros((n, n))

    for j in range(S):
        for k, ak in enumerate(a, start=1):
            if j + k <= S:
                A[j + k, j] = ak

    for j in range(1, S + 1):
        A[j - 1, j] = mu

    for i in range(n):
        A[i, i] = -np.sum(A[:, i])

    d = [1.5 ** i for i in range(S)]

    return A, d


def example_class3(lam=1.0, b=(2.0, 1.0), S=5):
    """
    Класс III: одиночные поступления, групповое обслуживание
    (группы размеров 1, 2). b_k >= b_{k+1}.
    """
    b = np.asarray(b, dtype=float)
    n = S + 1
    A = np.zeros((n, n))

    for j in range(S):
        A[j + 1, j] = lam

    for j in range(1, S + 1):
        for k, bk in enumerate(b, start=1):
            if j - k >= 0:
                A[j - k, j] = bk

    for i in range(n):
        A[i, i] = -np.sum(A[:, i])

    d = [1.5 ** i for i in range(S)]

    return A, d


def example_class4(a=(1.0, 0.5), b=(2.0, 1.0), S=5):
    """Класс IV: групповые поступления + групповое обслуживание."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = S + 1
    A = np.zeros((n, n))

    for j in range(S):
        for k, ak in enumerate(a, start=1):
            if j + k <= S:
                A[j + k, j] = ak

    for j in range(1, S + 1):
        for k, bk in enumerate(b, start=1):
            if j- k >= 0:
                A[j - k, j] = bk

    for i in range(n):
        A[i, i] = -np.sum(A[:, i])

    d = [1.5 ** i for i in range(S)]

    return A, d


def example_class5(lam=1.0, mu=2.0, S=5):
    """
    Класс V: ПРГ с поглощением в нуле (lambda_0 = 0, mu_1 > 0).
    """
    n = S + 1
    A = np.zeros((n, n))

    for j in range(1, S):
        A[j + 1, j] = lam
        A[j - 1, j] = mu

    A[S, S - 1] = lam
    A[S - 1, S] = mu

    for i in range(n):
        A[i, i] = -np.sum(A[:, i])

    delta = np.sqrt(mu / lam)
    d = [delta ** i for i in range(S)]

    return A, d


def example_class6(lam=1.0, mu=4.0, beta_cat=0.5, S=5):
    """
    Класс VI: ПРГ с катастрофами (из каждого состояния в 0
    с интенсивностью beta_cat).
    """
    n = S + 1
    A = np.zeros((n, n))

    for j in range(S):
        A[j + 1, j] = lam

    for j in range(1, S + 1):
        A[j - 1, j] = mu

    for j in range(1, S + 1):
        A[0, j] = beta_cat

    for i in range(n):
        A[i, i] = -np.sum(A[:, i])

    d = [1.0] * S

    return A, d
