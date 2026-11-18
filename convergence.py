"""
convergence.py
Моделирование прямой системы Колмогорова dp/dt = A p
и построение графика сходимости к стационарному распределению.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from examples import example_MM1

def get_stationary_distribution(A):

    """Находит стационарное распределение pi (решение A^T pi = 0, сумма pi = 1)."""

    n = A.shape[0]
    M = A.copy()

    M[-1, :] = 1.0

    b = np.zeros(n)
    b[-1] = 1.0

    pi = np.linalg.solve(M, b)
    return pi


def main():

    lam, mu, S = 1.0, 4.0, 5
    A, _ = example_MM1(lam=lam, mu=mu, S=S)

    p0 = np.zeros(S + 1)
    p0[0] = 1.0

    pi = get_stationary_distribution(A)

    def rhs(t, p):
        return A @ p

    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 200)

    sol = solve_ivp(rhs, t_span, p0, t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)

    diff = np.sum(np.abs(sol.y - pi[:, np.newaxis]), axis=0)

    plt.figure(figsize=(12, 5))

    # График 1 (сх-ть вер-тей)
    plt.subplot(1, 2, 1)
    for i in range(S + 1):
        plt.plot(sol.t, sol.y[i, :], label=f"p_{i}(t)")

        if i == 0:
            plt.axhline(y=pi[i], color='k', linestyle='--', alpha=0.7, label='Стационарные pi')
        else:
            plt.axhline(y=pi[i], color='k', linestyle='--', alpha=0.7)

    plt.xlabel("Время t")
    plt.ylabel("Вероятность")
    plt.title("Сходимость вероятностей (M/M/1, S=5)")
    plt.legend()
    plt.grid(True)

    # График 2 (логарифм нормы разности)
    plt.subplot(1, 2, 2)
    plt.semilogy(sol.t, diff, 'r-', linewidth=2, label="||p(t) - pi||_1")
    plt.xlabel("Время t")
    plt.ylabel("Норма разности (лог. шкала)")
    plt.title("Скорость сходимости к стационару")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("convergence_MM1.png", dpi=150)
    print("График сходимости сохранен: convergence_MM1.png")
    plt.show()

if __name__ == "__main__":
    main()

