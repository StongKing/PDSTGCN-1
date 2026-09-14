from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import configparser
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lib.metrics import masked_mae, masked_mse, pdstgcn_loss
from lib.utils import (
    compute_val_loss_mstgcn,
    get_adjacency_matrix,
    load_graphdata_channel1,
    predict_and_save_results_mstgcn,
)
from model.ASTGCN_r import make_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configurations/DIVVY_astgcn.conf")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--epoch", type=int, default=None, help="checkpoint epoch for --predict-only")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config["Data"]
    training_config = config["Training"]

    adj_filename = data_config["adj_filename"]
    graph_signal_matrix_filename = data_config["graph_signal_matrix_filename"]
    dynamic_graph_filename = data_config["dynamic_graph_filename"]
    id_filename = data_config.get("id_filename", fallback=None)
    num_of_vertices = int(data_config["num_of_vertices"])
    points_per_hour = int(data_config["points_per_hour"])
    num_for_predict = int(data_config["num_for_predict"])
    len_input = int(data_config["len_input"])
    dataset_name = data_config["dataset_name"]
    fleet_size = int(data_config["fleet_size"])

    ctx = training_config.get("ctx", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = ctx
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate = float(training_config["learning_rate"])
    epochs = int(training_config["epochs"])
    start_epoch = int(training_config["start_epoch"])
    batch_size = int(training_config["batch_size"])
    num_of_weeks = int(training_config["num_of_weeks"])
    num_of_days = int(training_config["num_of_days"])
    num_of_hours = int(training_config["num_of_hours"])
    time_strides = num_of_hours
    nb_chev_filter = int(training_config["nb_chev_filter"])
    nb_time_filter = int(training_config["nb_time_filter"])
    in_channels = int(training_config["in_channels"])
    nb_block = int(training_config["nb_block"])
    K = int(training_config["K"])
    loss_function = training_config["loss_function"].lower()
    metric_method = training_config["metric_method"].lower()
    missing_value = float(training_config["missing_value"])
    model_name = training_config["model_name"]
    seed = int(training_config.get("seed", 20260913))
    set_seed(seed)

    expected_input = points_per_hour * num_of_hours
    if expected_input != len_input:
        raise ValueError(f"len_input={len_input}, but points_per_hour*num_of_hours={expected_input}")
    if len_input != 6 or num_for_predict != 6:
        print(f"WARNING: this Divvy project was designed for 6->6; current config is {len_input}->{num_for_predict}")

    print("Read configuration file:", args.config)
    print("DEVICE:", device)
    print("nodes / fleet:", num_of_vertices, fleet_size)
    print("history / prediction:", len_input, num_for_predict)

    (
        train_loader,
        train_target_tensor,
        val_loader,
        val_target_tensor,
        test_loader,
        test_target_tensor,
        graph_store,
        mean,
        std,
    ) = load_graphdata_channel1(
        graph_signal_matrix_filename,
        dynamic_graph_filename,
        num_of_hours,
        num_of_days,
        num_of_weeks,
        device,
        batch_size,
        in_channels,
        shuffle=True,
    )

    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
    net = make_model(
        device,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        adj_mx,
        num_for_predict,
        len_input,
        num_of_vertices,
        fleet_size,
    )

    params_path = Path("experiments") / dataset_name / (
        f"{model_name}_h{num_of_hours}d{num_of_days}w{num_of_weeks}_"
        f"channel{in_channels}_{learning_rate:.6e}"
    )

    if loss_function == "masked_mse":
        criterion = masked_mse
        masked_flag = True
    elif loss_function == "masked_mae":
        criterion = masked_mae
        masked_flag = True
    elif loss_function == "mae":
        criterion = nn.L1Loss().to(device)
        masked_flag = False
    elif loss_function in {"mse", "rmse"}:
        criterion = nn.MSELoss().to(device)
        masked_flag = False
    else:
        raise ValueError(f"Unknown loss_function={loss_function}")

    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)

    def checkpoint(epoch):
        return params_path / f"epoch_{epoch}.params"

    if args.predict_only:
        if args.epoch is None:
            files = sorted(params_path.glob("epoch_*.params"), key=lambda p: int(p.stem.split("_")[-1]))
            if not files:
                raise FileNotFoundError(f"No checkpoints in {params_path}")
            ckpt = files[-1]
            epoch = int(ckpt.stem.split("_")[-1])
        else:
            epoch = args.epoch
            ckpt = checkpoint(epoch)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        predict_and_save_results_mstgcn(
            net, test_loader, test_target_tensor, graph_store, epoch, str(params_path), "test", device
        )
        return

    if start_epoch == 0:
        if params_path.exists():
            shutil.rmtree(params_path)
        params_path.mkdir(parents=True, exist_ok=True)
    else:
        if not params_path.exists():
            raise FileNotFoundError(params_path)
        net.load_state_dict(torch.load(checkpoint(start_epoch), map_location=device))

    print("params_path:", params_path)
    print("batch size:", batch_size, "K:", K, "filters:", nb_chev_filter, nb_time_filter)
    print("loss:", loss_function, "metric:", metric_method)
    print("IMPORTANT: metric_method=unmask is correct for inventory because zero bikes is a valid observation.")

    best_val = np.inf
    best_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        val_loss = compute_val_loss_mstgcn(
            net, val_loader, graph_store, criterion, masked_flag, missing_value, device
        )
        print(f"Epoch {epoch:03d} pre-train val loss: {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), checkpoint(epoch))

        net.train()
        train_losses = []
        for encoder_inputs, labels, graph_idx in train_loader:
            encoder_inputs = encoder_inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            A = graph_store.dense(graph_idx.numpy(), device)

            optimizer.zero_grad(set_to_none=True)
            outputs = net(encoder_inputs, A, apply_rounding=False)
            loss = pdstgcn_loss(outputs, labels, alpha=0.005, beta=0.01)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        print(
            f"Epoch {epoch:03d} train loss: {np.mean(train_losses):.6f} | "
            f"best val: {best_val:.6f} @ {best_epoch}"
        )

    # Re-evaluate the best checkpoint after the final epoch.
    net.load_state_dict(torch.load(checkpoint(best_epoch), map_location=device))
    predict_and_save_results_mstgcn(
        net, test_loader, test_target_tensor, graph_store, best_epoch, str(params_path), "test", device
    )


if __name__ == "__main__":
    main()
