import wandb
import logging
from argparse import ArgumentParser
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from experiments.toy.problem import Toy
from experiments.toy.problem_transformed import Toy_Transformed
from experiments.toy.utils import plot_2d_pareto
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    set_logger,
)
from methods.gradient_weight_methods import GradientWeightMethods
import os

set_logger()


def main(method_type, device, n_iter, scale, transform):
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    n_tasks = 2
    print("Weight method parameters: ", transform, scale)
    if transform:
        F = Toy_Transformed(scale=scale)
    else:
        F = Toy(scale=scale)

    all_traj = dict()

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([0.1, 0.1]),
        torch.Tensor([9.0, 9.0]),
        torch.Tensor([-7.5, -0.5]),
        torch.Tensor([9, -1.0]),
    ]

    for i, init in enumerate(inits):
        traj = []
        x = init.clone()
        x.requires_grad = True
        x = x.to(device)

        method = GradientWeightMethods(
            method=method_type,
            device=device,
            n_tasks=n_tasks,
            **weight_methods_parameters[method_type],
        )

        optimizer = torch.optim.Adam(
            [
                dict(params=[x], lr=1e-3),
                dict(params=method.parameters(), lr=args.method_params_lr),
            ],
        )

        for _ in tqdm(range(n_iter)):
            traj.append(x.cpu().detach().numpy().copy())
            
            optimizer.zero_grad()
            f = F(x, False)
            loss_fns = [lambda x_: F(x_, False)[i] for i in range(n_tasks)]
            _ = method.backward(
                losses=f,
                shared_parameters=(x,),
                task_specific_parameters=None,
                last_shared_parameters=None,
                representation=None
            )
            optimizer.step()

        all_traj[i] = dict(init=init.cpu().detach().numpy().copy(), traj=np.array(traj))

    return all_traj


if __name__ == "__main__":
    parser = ArgumentParser(
        "Toy example (modification of the one in CAGrad)", parents=[common_parser]
    )
    parser.set_defaults(n_epochs=35000, gradient_method="dibsmtl", data_path=None)
    parser.add_argument(
        "--scale", default=1.0, type=float, help="scale for first loss"
    )
    parser.add_argument("--out-path", default="outputs", type=Path, help="output path")
    parser.add_argument("--transform", action="store_true", help="Apply loss transform")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    args = parser.parse_args()
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    out_path = args.out_path
    out_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Logs and plots are saved in: {out_path.as_posix()}")
    print("transform_args: ", args.transform)
    device = torch.device("cpu")
    all_traj = main(
        method_type=args.gradient_method, device=device, n_iter=args.n_epochs, scale=args.scale, transform=args.transform,
    )

    # store trajectories as pickle (transformed separately)
    if args.transform:
        with open(out_path / "transformed" / f"{args.gradient_method}_trajectories.pkl", "wb") as f:
            pickle.dump(all_traj, f)
    else:
        with open(out_path / "nominal" / f"{args.gradient_method}_trajectories.pkl", "wb") as f:
            pickle.dump(all_traj, f)

    ax, fig = plot_2d_pareto(trajectories=all_traj, scale=args.scale, transform=False)

    title_map = {
        "nashmtl": "Nash-MTL",
        "cagrad": "CAGrad",
        "mgda": "MGDA",
        "pcgrad": "PCGrad",
        "ls": "LS",
        "famo": "FAMO",
        "dibsmtl": "1-Step DiBS-MTL",
        "dibsmtl_multi_step": "5-Step DiBS-MTL",
        "imtl": "IMTL-G",
        "uw": "UW",
        "fairgrad": "FairGrad",
        "gradnorm": "GradNorm",
    }
    ax.set_title(title_map[args.gradient_method], fontsize=45)
    plt.savefig(
        out_path / f"{args.gradient_method}.png",
        # bbox_extra_artists=(legend,),
        bbox_inches="tight",
        facecolor="white",
    )

    ax.set_title(title_map[args.gradient_method], fontsize=45)

    # ensure subdir exists
    subdir = out_path / ("transformed" if args.transform else "nominal")
    print(subdir)
    subdir.mkdir(parents=True, exist_ok=True)

    png_path = subdir / f"{args.gradient_method}.png"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    # tikzplotlib.save(str(tex_path))
    plt.close(fig)

    if args.transform:
        ax_t, fig_t = plot_2d_pareto(trajectories=all_traj, scale=1.0, transform=True)
        ax_t.set_title(title_map[args.gradient_method], fontsize=45)
        scaled_png = subdir / f"{args.gradient_method}_scaled.png"
        scaled_tex = subdir / f"{args.gradient_method}_scaled.tex"
        fig_t.tight_layout()
        fig_t.savefig(scaled_png, dpi=300)
        # tikzplotlib.save(str(scaled_tex))
        plt.close(fig_t)

    if wandb.run is not None:
        wandb.log({"Pareto Front": wandb.Image((out_path / f"{args.gradient_method}.png").as_posix())})

        wandb.finish()