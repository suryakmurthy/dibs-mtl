import matplotlib
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from experiments.toy.problem import Toy
from experiments.toy.problem_transformed import Toy_Transformed

def plot_2d_pareto(trajectories: dict, scale, transform=False):
    """Adaptation of code from: https://github.com/Cranial-XIX/CAGrad"""
    print("Plotting 2D Pareto front...")
    fig, ax = plt.subplots(figsize=(6, 5))
    if transform:
        F = Toy_Transformed(scale=scale)
    else:
        F = Toy(scale=scale)
    losses = [F.batch_forward(torch.from_numpy(res["traj"])) for res in trajectories.values()]

    # Plot the Pareto front
    yy = -8.3552
    x = np.linspace(-7, 7, 1000)
    Xs = torch.from_numpy(np.stack((x, [yy] * len(x))).T).double()
    Ys = F.batch_forward(Xs).numpy()
    ax.plot(Ys[:, 0], Ys[:, 1], "-", linewidth=8, color="#72727A", label="Pareto Front")

    # Plot trajectories
    for i, tt in enumerate(losses):
        ax.scatter(
            tt[0, 0], tt[0, 1],
            color="k", s=150, zorder=10,
            label="Initial Point" if i == 0 else None,
        )
        colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, tt.shape[0]))
        ax.scatter(tt[:, 0], tt[:, 1], color=colors, s=5, zorder=9)

    # Styling
    sns.despine()
    # ax.tick_params(axis="both", which="major", labelsize=45)
    ax.set_xlabel(r"$L_1$", fontsize=45)
    ax.set_ylabel(r"$L_2$", fontsize=45)
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)

    # Updated tick fontsize (modern Matplotlib)
    for label in ax.get_xticklabels():
        label.set_fontsize(20)
        label.set_rotation(90)
    for label in ax.get_yticklabels():
        label.set_fontsize(20)
    # ax.set_xticks(np.linspace(-15, 0, 6))  # Customize spacing

    plt.tight_layout()

    # legend = ax.legend(
    #     loc=2, bbox_to_anchor=(-0.15, 1.3),
    #     frameon=False, fontsize=20, ncol=2
    # )

    return ax, fig #, legend