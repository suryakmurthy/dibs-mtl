import logging
import wandb
from argparse import ArgumentParser
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.nyuv2.data import NYUv2
from experiments.nyuv2.models import SegNet, SegNetMtan
from experiments.nyuv2.utils import ConfMatrix, delta_fn, depth_error, normal_error
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device
    # print("Checking input data type: ", x_pred, x_output, task_type)
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss

def get_param_snapshot(model):
    """Returns a dictionary mapping param names to a clone of their values."""
    return {name: param.clone().detach() for name, param in model.named_parameters()}

def compare_param_snapshots(snapshot_before, snapshot_after, atol=1e-6):
    """Prints which parameters changed between two snapshots."""
    changed = []
    unchanged = []
    for name in snapshot_before:
        if not torch.allclose(snapshot_before[name], snapshot_after[name], atol=atol):
            changed.append(name)
        else:
            unchanged.append(name)
    return changed, unchanged


def main(path, lr, bs, ss, device):
    # ----
    # Nets
    # ---
    os.makedirs(f"checkpoints", exist_ok=True)
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)
    step_size = ss
    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_set = NYUv2(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    nyuv2_test_set = NYUv2(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=bs, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=bs, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method]
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=weight_method.parameters(), lr=args.method_params_lr),
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) #TODO: implement for dibs as well

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    for epoch in epoch_iter:
        cost = np.zeros(24, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                    calc_loss(train_pred[2], train_normal, "normal"),
                )
            )

            if args.method == "dibsmtl" or args.method == "ms_dibsmtl" or args.method == "ss_dibsmtl":
                # define losses as functions
                # loss_fns = [
                #     lambda x_, i=i: calc_loss(
                #         model(x_)[0][i], train_label if i == 0 else (train_depth if i == 1 else train_normal),
                #         ["semantic","depth","normal"][i]
                #     )
                #     for i in range(3)
                # ]
                # compute DiBS-MTL update direction
                loss, extra_outputs = weight_method.backward(
                    losses=losses,
                    shared_parameters=list(model.shared_parameters()),
                    task_specific_parameters=list(model.task_specific_parameters()),
                    last_shared_parameters=list(model.last_shared_parameters()),
                    representation=features,
                    learning_rate = scheduler.get_last_lr()[0],
                    step_size = step_size,
                )
                # param_before = get_param_snapshot(model)
                optimizer.step()

                # x_change = weight_method.method.dibs_update( # TODO: check if this is correct
                #     losses=losses,
                #     shared_parameters=list(model.parameters()),
                #     radius=scheduler.get_last_lr()[0],
                #     max_steps=100,
                #     step_size=scheduler.get_last_lr()[0] / 10,
                # )

                # # manually apply the step
                # param_before = get_param_snapshot(model)
                # with torch.no_grad():
                #     for param, delta in zip(model.parameters(), x_change):
                #         param.add_(scheduler.get_last_lr()[0] * delta)  # scale the step like in the toy example TODO: make lr changes for dibs here
                # param_after = get_param_snapshot(model)
                # changed, unchanged = compare_param_snapshots(param_before, param_after)

                # print("Changed parameters:")
                # for name in changed:
                #     print("  ", name)

                # print("Unchanged parameters:")
                # for name in unchanged:
                #     print("  ", name)
            else:
                # do the regular backward and optimizer.step() for other methods
                loss, extra_outputs = weight_method.backward(
                    losses=losses,
                    shared_parameters=list(model.shared_parameters()),
                    task_specific_parameters=list(model.task_specific_parameters()),
                    last_shared_parameters=list(model.last_shared_parameters()),
                    representation=features,
                )
                # param_before = get_param_snapshot(model)
                optimizer.step()
                # param_after = get_param_snapshot(model)
                # changed, unchanged = compare_param_snapshots(param_before, param_after)

                # print("Changed parameters:")
                # for name in changed:
                #     print("  ", name)

                # print("Unchanged parameters:")
                # for name in unchanged:
                #     print("  ", name)


            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
                train_pred[2], train_normal
            )
            avg_cost[epoch, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"normal loss: {losses[2].item():.3f}"
            )

        # scheduler
        scheduler.step() #TODO: check if we can use this for dibs
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset) #test_dataset.next()#.next is deprecated
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal
                )
                avg_cost[epoch, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
            )

            # print results
            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
                f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m (test)"
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"{avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} || "
                f"TEST: {avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]:.4f} {avg_cost[epoch, 14]:.4f} | "
                f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | {avg_cost[epoch, 18]:.4f} "
                f"{avg_cost[epoch, 19]:.4f} {avg_cost[epoch, 20]:.4f} {avg_cost[epoch, 21]:.4f} {avg_cost[epoch, 22]:.4f} {avg_cost[epoch, 23]:.4f} "
                f"| {test_delta_m:.3f}"
            )
            # Save model checkpoint
            if epoch % 50 == 0 or epoch == epochs - 1:
                save_path = f"checkpoints/{args.model}_{args.method}_epoch{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'weight_method_state_dict': weight_method.state_dict() if hasattr(weight_method, 'state_dict') else None,
                    'avg_cost': avg_cost[epoch],
                    'args': vars(args)  # <-- Add this
                }, save_path)

            if wandb.run is not None:
                # Create dictionary to log
                metrics = {
                    "Train Semantic Loss": avg_cost[epoch, 0],
                    "Train Mean IoU": avg_cost[epoch, 1],
                    "Train Pixel Accuracy": avg_cost[epoch, 2],
                    "Train Depth Loss": avg_cost[epoch, 3],
                    "Train Absolute Error": avg_cost[epoch, 4],
                    "Train Relative Error": avg_cost[epoch, 5],
                    "Train Normal Loss": avg_cost[epoch, 6],
                    "Train Loss Mean": avg_cost[epoch, 7],
                    "Train Loss Med": avg_cost[epoch, 8],
                    "Train Loss <11.25": avg_cost[epoch, 9],
                    "Train Loss <22.5": avg_cost[epoch, 10],
                    "Train Loss <30": avg_cost[epoch, 11],
                    "Test Semantic Loss": avg_cost[epoch, 12],
                    "Test Mean IoU": avg_cost[epoch, 13],
                    "Test Pixel Accuracy": avg_cost[epoch, 14],
                    "Test Depth Loss": avg_cost[epoch, 15],
                    "Test Absolute Error": avg_cost[epoch, 16],
                    "Test Relative Error": avg_cost[epoch, 17],
                    "Test Normal Loss": avg_cost[epoch, 18],
                    "Test Loss Mean": avg_cost[epoch, 19],
                    "Test Loss Med": avg_cost[epoch, 20],
                    "Test Loss <11.25": avg_cost[epoch, 21],
                    "Test Loss <22.5": avg_cost[epoch, 22],
                    "Test Loss <30": avg_cost[epoch, 23],
                    "Test ∆m": test_delta_m,
                }
                wandb.log(metrics, step=epoch)
                # wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                # wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                # wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                # wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                # wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                # wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)
                # wandb.log({"Train Normal Loss": avg_cost[epoch, 6]}, step=epoch)
                # wandb.log({"Train Loss Mean": avg_cost[epoch, 7]}, step=epoch)
                # wandb.log({"Train Loss Med": avg_cost[epoch, 8]}, step=epoch)
                # wandb.log({"Train Loss <11.25": avg_cost[epoch, 9]}, step=epoch)
                # wandb.log({"Train Loss <22.5": avg_cost[epoch, 10]}, step=epoch)
                # wandb.log({"Train Loss <30": avg_cost[epoch, 11]}, step=epoch)

                # wandb.log({"Test Semantic Loss": avg_cost[epoch, 12]}, step=epoch)
                # wandb.log({"Test Mean IoU": avg_cost[epoch, 13]}, step=epoch)
                # wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 14]}, step=epoch)
                # wandb.log({"Test Depth Loss": avg_cost[epoch, 15]}, step=epoch)
                # wandb.log({"Test Absolute Error": avg_cost[epoch, 16]}, step=epoch)
                # wandb.log({"Test Relative Error": avg_cost[epoch, 17]}, step=epoch)
                # wandb.log({"Test Normal Loss": avg_cost[epoch, 18]}, step=epoch)
                # wandb.log({"Test Loss Mean": avg_cost[epoch, 19]}, step=epoch)
                # wandb.log({"Test Loss Med": avg_cost[epoch, 20]}, step=epoch)
                # wandb.log({"Test Loss <11.25": avg_cost[epoch, 21]}, step=epoch)
                # wandb.log({"Test Loss <22.5": avg_cost[epoch, 22]}, step=epoch)
                # wandb.log({"Test Loss <30": avg_cost[epoch, 23]}, step=epoch)
                # wandb.log({"Test ∆m": test_delta_m}, step=epoch)


if __name__ == "__main__":
    parser = ArgumentParser("NYUv2", parents=[common_parser])
    parser.set_defaults(
        data_path="dataset",
        lr=1e-4,
        n_epochs=200,
        batch_size=2,
    )
    parser.add_argument("--model", type=str, default="mtan", choices=["segnet", "mtan"])
    parser.add_argument("--apply-augmentation", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)

    if args.wandb_project is not None:
        wandb.init(project="dibsmtl-sweep")
        
        config_path = "experiments/nyuv2/sweep.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding='utf-8') as f:
                config = yaml.safe_load(f)
        step_size = config.step_size  # use sweep value
        lr = config.lr  # could sweep this too later
        batch_size = config.batch_size
        sweep_id = wandb.sweep(sweep=config, project="dibsmtl-sweep")
        if not args.sweep_id is None:
            sweep_id = args.sweep_id
            print(f"Using provided sweep ID: {sweep_id}")
        wandb.agent(sweep_id, function=main, count=1)
    else:
        step_size = args.lr / 10
        lr = args.lr
        batch_size = args.batch_size

    # device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=lr, bs=batch_size, ss=step_size, device=device)

    if wandb.run is not None:
        wandb.finish()