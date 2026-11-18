"""
demo.py
Демонстрация работы пакета markov_tools на примерах классов I-VI.
"""

import numpy as np
from markov_tools import (
    analyze, print_matrix, build_D, build_D_inv,
    compute_B, compute_B_star, log_norm, is_essentially_nonneg
)
from examples import (
    example_MM1, example_MMNN, find_beta_MMNN_exact,
    example_class2, example_class3, example_class4,
    example_class5, example_class6,
)


def demo_full_pipeline(A, d, name, expected_beta=None):
    """Полный конвейер с печатью всех матриц."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    print_matrix(A, "A (исходная)")
    print(f"\nd = {d}")

    result = analyze(A, d, verbose=False)

    print_matrix(result['B'], "B = A[1:,1:] - A[1:,0]")
    print_matrix(result['D'], "D (1.26)")
    print_matrix(result['D_inv'], "D^{-1}")
    print_matrix(result['B_star'], "B** = D B D^{-1}")

    print(f"\nγ(B**) = {result['gamma']:.6f}")
    print(f"β = -γ = {result['beta']:.6f}")
    print(f"B** существенно неотрицательна: {result['ess_nonneg']}")

    if expected_beta is not None:
        err = abs(result['beta'] - expected_beta)
        print(f"\nТеоретическое β = {expected_beta:.6f}")
        print(f"Ошибка = {err:.6f}")

    return result


if __name__ == "__main__":
    # ---------- Класс I: M/M/1 ----------
    lam, mu, S = 1.0, 4.0, 5
    A, d = example_MM1(lam=lam, mu=mu, S=S)
    beta_exact = lam + mu - 2 * np.sqrt(lam * mu) * np.cos(np.pi / (S + 1))
    result = demo_full_pipeline(A, d,
                                "M/M/1, λ=1, μ=4, S=5 (класс I)",
                                expected_beta=beta_exact)

    # ---------- Класс I: M/M/N/N ----------
    lam, mu, N = 1.0, 1.0, 3
    A, d = example_MMNN(lam=lam, mu=mu, N=N)
    beta_exact = find_beta_MMNN_exact(lam=lam, mu=mu, N=N)
    result = demo_full_pipeline(A, d,
                                f"M/M/{N}/{N}, λ={lam}, μ={mu} (класс I)",
                                expected_beta=beta_exact)

    # ---------- Класс II ----------
    A, d = example_class2(a=(1.0, 0.5), mu=1.0, S=5)
    demo_full_pipeline(A, d, "Класс II: групповые поступления")

    # ---------- Класс III ----------
    A, d = example_class3(lam=1.0, b=(2.0, 1.0), S=5)
    demo_full_pipeline(A, d, "Класс III: групповое обслуживание")

    # ---------- Класс IV ----------
    A, d = example_class4(a=(1.0, 0.5), b=(2.0, 1.0), S=5)
    demo_full_pipeline(A, d, "Класс IV: оба групповых")

    # ---------- Класс V ----------
    A, d = example_class5(lam=1.0, mu=2.0, S=5)
    demo_full_pipeline(A, d, "Класс V: поглощение в нуле")

    # ---------- Класс VI (специальный) ----------
    A, d = example_class6(lam=1.0, mu=4.0, beta_cat=0.5, S=5)
    print(f"\n{'=' * 70}")
    print("  Класс VI: катастрофы")
    print(f"{'=' * 70}")

    beta_star = 0.5
    A_star = A.copy()
    A_star[0, :] -= beta_star

    gamma_A_star = log_norm(A_star)

    print(f"\nγ(A*) = {gamma_A_star:.6f}  (теория: -β_* = {-beta_star:.6f})")
    print(f"β_* = -γ(A*) = {-gamma_A_star:.6f}")
