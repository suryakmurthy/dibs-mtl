import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

import sys
sys.path.append("../..")
from experiments.cityscapes.data import Cityscapes
from experiments.cityscapes.models import SegNet, SegNetMtan
from experiments.cityscapes.utils import ConfMatrix, delta_fn, depth_error
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)

from methods.loss_weight_methods import LossWeightMethods
from methods.gradient_weight_methods import GradientWeightMethods

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space (used for depth)
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross-entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    elif task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return loss

def main(path, lr, bs, device):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/{timestr}_{args.loss_method}_{args.gradient_method}_seed{args.seed}_log.txt"

    # Nets
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)
    print(args.loss_method, args.gradient_method)
    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on CityScapes."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    cityscapes_train_set = Cityscapes(root=path.as_posix(), train=True, augmentation=args.apply_augmentation)
    cityscapes_val_set = Cityscapes(root=path.as_posix(), train=False, augmentation=False)
    cityscapes_test_set = Cityscapes(root=path.as_posix(), train=False, augmentation=False)

    train_loader = DataLoader(dataset=cityscapes_train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset=cityscapes_val_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(dataset=cityscapes_test_set, batch_size=bs, shuffle=False)
    # loss_weight method
    loss_weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    print(args.loss_method)
    loss_weight_method = LossWeightMethods(
        args.loss_method, n_tasks=2, device=device, **loss_weight_methods_parameters[args.loss_method]
    )

    # gradient_weight method
    gradient_weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    gradient_weight_method = GradientWeightMethods(
        args.gradient_method, n_tasks=2, device=device, **gradient_weight_methods_parameters[args.gradient_method]
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=loss_weight_method.parameters(), lr=args.method_params_lr),
            dict(params=gradient_weight_method.parameters(), lr=args.method_params_lr),
        ],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    val_batch = len(val_loader)
    test_batch = len(test_loader)
    # now only two tasks: semantic and depth -> 6 train metrics + 6 val metrics = 12
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)

    # best model to test
    best_epoch = None
    best_eval = 0

    # print result head
    print(
        f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
        f"∆m"
    )

    train_time_sum = 0.0

    # train batch for IGBv2
    if args.loss_method == 'igbv2':
        loss_weight_method.method.train_batch = train_batch

    for epoch in epoch_iter:
        cost = np.zeros(12, dtype=np.float32)
        conf_mat = ConfMatrix(model.segnet.class_nb)
        avg_loss_weights = torch.zeros(2).to(device)

        start_train_time = time.time()

        # reward scale for IGBv2
        if args.loss_method == 'igbv2':
            loss_weight_method.method.reward_scale = lr / optimizer.param_groups[0]['lr']

        for j, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth = train_depth.to(device)

            train_pred, features = model(train_data, return_representation=True)

            # only semantic and depth losses
            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                )
            )

            weighted_losses, loss_weights = loss_weight_method.get_weighted_losses(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )
            avg_loss_weights += loss_weights['weights'] / train_batch

            loss, gradient_weights = gradient_weight_method.backward(
                losses=weighted_losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            # TRAIN metrics (0..5)
            cost[0] = losses[0].item()  # semantic loss
            # (1,2) will be filled from conf_mat metrics after the epoch
            cost[3] = losses[1].item()  # depth loss
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            epoch_iter.set_description(
                f"[{epoch} {j + 1}/{train_batch}] losses: {losses[0].item():.3f} {losses[1].item():.3f} "
                f"weights: {loss_weights['weights'][0].item():.3f} {loss_weights['weights'][1].item():.3f}"
            )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # base_losses for IGBv1 and IGBv2 (semantic, depth)
        if 'igb' in args.loss_method and epoch == args.base_epoch:
            base_losses = torch.Tensor(avg_cost[epoch, [0, 3]]).to(device)
            loss_weight_method.method.base_losses = base_losses

        end_train_time = time.time()
        train_time_sum += end_train_time - start_train_time

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            for j, batch in enumerate(val_loader):
                val_data, val_label, val_depth = batch
                val_data, val_label = val_data.to(device), val_label.long().to(device)
                val_depth = val_depth.to(device)

                val_pred = model(val_data)
                val_loss = torch.stack(
                    (
                        calc_loss(val_pred[0], val_label, "semantic"),
                        calc_loss(val_pred[1], val_depth, "depth"),
                    )
                )

                conf_mat.update(val_pred[0].argmax(1).flatten(), val_label.flatten())

                # VAL metrics are stored in avg_cost columns 6..11
                cost[6] = val_loss[0].item()
                cost[9] = val_loss[1].item()
                cost[10], cost[11] = depth_error(val_pred[1], val_depth)
                avg_cost[epoch, 6:] += cost[6:] / val_batch

            # compute mIoU and acc for validation
            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            # Val Delta_m (mean IoU, pix acc, abs err, rel err)
            val_delta_m = delta_fn(avg_cost[epoch, [7, 8, 10, 11]])

        # Use val_delta_m as evaluation metric for all methods
        eval_value = val_delta_m

        results = (
            f"Epoch: {epoch:04d}\n"
            f"AVERAGE LOSS WEIGHTS: {avg_loss_weights[0]:.4f} {avg_loss_weights[1]:.4f}\n"
            f"TRAIN: {avg_cost[epoch,0]:.4f} {avg_cost[epoch,1]:.4f} {avg_cost[epoch,2]:.4f} | "
            f"{avg_cost[epoch,3]:.4f} {avg_cost[epoch,4]:.4f} {avg_cost[epoch,5]:.4f}\n"
            f"VAL:   {avg_cost[epoch,6]:.4f} {avg_cost[epoch,7]:.4f} {avg_cost[epoch,8]:.4f} | "
            f"{avg_cost[epoch,9]:.4f} {avg_cost[epoch,10]:.4f} {avg_cost[epoch,11]:.4f} | {val_delta_m:.3f}\n"
        )

        if best_epoch is None or eval_value < best_eval:
            best_epoch = epoch
            best_eval = eval_value

            # test
            test_cost = np.zeros(6, dtype=np.float32)
            test_avg_cost = np.zeros(6, dtype=np.float32)
            conf_mat = ConfMatrix(model.segnet.class_nb)
            with torch.no_grad():
                for j, batch in enumerate(test_loader):
                    test_data, test_label, test_depth = batch
                    test_data, test_label = test_data.to(device), test_label.long().to(device)
                    test_depth = test_depth.to(device)

                    test_pred = model(test_data)
                    test_loss = torch.stack(
                        (
                            calc_loss(test_pred[0], test_label, "semantic"),
                            calc_loss(test_pred[1], test_depth, "depth"),
                        )
                    )

                    conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                    test_cost[0] = test_loss[0].item()
                    test_cost[3] = test_loss[1].item()
                    test_cost[4], test_cost[5] = depth_error(test_pred[1], test_depth)
                    test_avg_cost += test_cost / test_batch

                # compute mIoU and acc
                test_avg_cost[1:3] = conf_mat.get_metrics()

                # Test Delta_m (mean IoU, pix acc, abs err, rel err)
                test_delta_m = delta_fn(test_avg_cost[[1, 2, 4, 5]])

            test_result = (
                f"TEST: {test_avg_cost[0]:.4f} {test_avg_cost[1]:.4f} {test_avg_cost[2]:.4f} | "
                f"{test_avg_cost[3]:.4f} {test_avg_cost[4]:.4f} {test_avg_cost[5]:.4f} | {test_delta_m:.3f}\n"
            )
            results += test_result
            # print test result
            print(test_result, end='')
        with open(log_file, mode="a") as log_f:
            log_f.write(results)

    train_time_log = f"Training time: {int(train_time_sum)}s\n"
    print(train_time_log, end='')
    with open(log_file, mode="a") as log_f:
        log_f.write(train_time_log)


if __name__ == "__main__":
    parser = ArgumentParser("CITYSCAPES", parents=[common_parser])
    parser.set_defaults(
        data_path="./dataset",
        lr=1e-4,
        n_epochs=500,
        batch_size=2,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="segnet",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument(
        "--apply-augmentation",
        type=str2bool,
        default=True,
        help="data augmentations"
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)
