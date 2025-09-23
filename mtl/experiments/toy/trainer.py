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
from methods.weight_methods import WeightMethods
import tikzplotlib

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

        method = WeightMethods(
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
    parser.set_defaults(n_epochs=35000, method="nashmtl", data_path=None)
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
        method_type=args.method, device=device, n_iter=args.n_epochs, scale=args.scale, transform=args.transform,
    )

    ax_n, fi_n = plot_2d_pareto(trajectories=all_traj, scale=args.scale, transform=False)

    title_map = {
        "nashmtl": "Nash-MTL",
        "cagrad": "CAGrad",
        "mgda": "MGDA",
        "pcgrad": "PCGrad",
        "ls": "LS",
        "dibsmtl": "DiBS-MTL",
        "ms_dibsmtl": "Multi-Step DiBS-MTL",
        "famo": "FAMO",
        "uw": "UW"
    }

    ax_n.set_title(title_map[args.method], fontsize=25)
    if args.transform:
        plt.savefig(
            out_path / f"transformed/{args.method}.pdf",
        )
        tikzplotlib.save(out_path / f"transformed/{args.method}.tex")

    else:
        plt.savefig(
            out_path / f"nominal/{args.method}.pdf",
        )
        tikzplotlib.save(out_path / f"nominal/{args.method}.tex")

    plt.close()
    if args.transform:
        ax_t, fi_t = plot_2d_pareto(trajectories=all_traj, scale=1.0, transform=True)
        ax_t.set_title(title_map[args.method], fontsize=25)
        plt.savefig(
            out_path / f"transformed/{args.method}_scaled.pdf",

        )
        tikzplotlib.save(out_path / f"transformed/{args.method}_scaled.tex")
        plt.close()

    if wandb.run is not None:
        wandb.log({"Pareto Front": wandb.Image((out_path / f"{args.method}.pdf").as_posix())})

        wandb.finish()