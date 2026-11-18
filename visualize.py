"""
visualize.py
Визуализация матриц (тепловые карты).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_matrix(M, title, ax=None, fmt=".2f"):
    """Тепловая карта матрицы с подписями элементов."""
    M = np.asarray(M, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Симметричная цветовая шкала вокруг 0
    vmax = max(abs(M.min()), abs(M.max()))
    im = ax.imshow(M, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(j, i, f"{val:{fmt}}",
                    ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_pipeline(result, savepath=None):
    """
    Построить сетку 2x3 с матрицами A, B, D, D^{-1}, B**.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    plot_matrix(result['A'], "A", ax=axes[0, 0])
    plot_matrix(result['B'], "B", ax=axes[0, 1])
    plot_matrix(result['D'], "D", ax=axes[0, 2])
    plot_matrix(result['D_inv'], "D⁻¹", ax=axes[1, 0])
    plot_matrix(result['B_star'], "B** = D B D⁻¹", ax=axes[1, 1])
    axes[1, 2].axis("off")
    axes[1, 2].text(0.5, 0.5,
                    f"γ(B**) = {result['gamma']:.4f}\n"
                    f"β = -γ = {result['beta']:.4f}\n"
                    f"Сущ. неотр.: {result['ess_nonneg']}",
                    ha="center", va="center", fontsize=14,
                    transform=axes[1, 2].transAxes)

    plt.suptitle("Конвейер преобразований", fontsize=14)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {savepath}")

    return fig