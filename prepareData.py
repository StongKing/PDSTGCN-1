# -*- coding: utf-8 -*-
"""Prepare chronological 6->6 Divvy samples for PDST-GCN.

This keeps the current repository data contract:
    X: (B, N, F, T_in)
    Y: (B, N, T_out)
and normalizes X with TRAINING statistics only.  Y remains in bike-count units.

For the Divvy configuration:
    points_per_hour = 6
    num_of_hours    = 1
    len_input       = 6
    num_for_predict = 6
therefore each valid sample consumes 12 consecutive states and the number of
usable sample starts is T - (6 + 6) + 1 = T - 11.  No hard-coded "23" is used.
"""

from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path

import numpy as np


def normalization(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return {"_mean": mean, "_std": std}, (train - mean) / std, (val - mean) / std, (test - mean) / std


def build_samples(data: np.ndarray, input_steps: int, pred_steps: int):
    T, N, F = data.shape
    ns = T - input_steps - pred_steps + 1
    if ns <= 0:
        raise ValueError(f"Not enough timesteps: T={T}, input={input_steps}, pred={pred_steps}")

    X = np.empty((ns, N, F, input_steps), dtype=np.float32)
    Y = np.empty((ns, N, pred_steps), dtype=np.float32)
    label_start = np.empty((ns, 1), dtype=np.int64)
    graph_id = np.arange(ns, dtype=np.int64).reshape(-1, 1)

    for g in range(ns):
        k = g + input_steps
        X[g] = data[g:k].transpose(1, 2, 0)
        Y[g] = data[k:k + pred_steps, :, 0].transpose(1, 0)
        label_start[g, 0] = k

    return X, Y, label_start, graph_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configurations/DIVVY_astgcn.conf")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    if "Data" not in config or "Training" not in config:
        raise ValueError(f"Invalid config: {args.config}")

    dcfg = config["Data"]
    tcfg = config["Training"]
    graph_signal_matrix_filename = dcfg["graph_signal_matrix_filename"]
    points_per_hour = int(dcfg["points_per_hour"])
    num_for_predict = int(dcfg["num_for_predict"])
    len_input = int(dcfg["len_input"])
    num_of_hours = int(tcfg["num_of_hours"])
    num_of_days = int(tcfg["num_of_days"])
    num_of_weeks = int(tcfg["num_of_weeks"])

    if num_of_days != 0 or num_of_weeks != 0:
        raise NotImplementedError(
            "PDSTGCN-1 Divvy configuration intentionally uses only the recent one-hour component: "
            "num_of_days=num_of_weeks=0."
        )

    input_steps = points_per_hour * num_of_hours
    if len_input != input_steps:
        raise ValueError(
            f"len_input={len_input} but points_per_hour*num_of_hours={input_steps}. "
            "For Divvy they must both equal 6."
        )

    raw = np.load(graph_signal_matrix_filename, allow_pickle=False)
    data = raw["data"].astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (T,N,F); got {data.shape}")

    X, Y, timestamp, graph_id = build_samples(data, input_steps, num_for_predict)
    ns = X.shape[0]
    split1 = int(ns * 0.60)
    split2 = int(ns * 0.80)

    train_x, val_x, test_x = X[:split1], X[split1:split2], X[split2:]
    train_y, val_y, test_y = Y[:split1], Y[split1:split2], Y[split2:]
    train_ts, val_ts, test_ts = timestamp[:split1], timestamp[split1:split2], timestamp[split2:]
    train_gid, val_gid, test_gid = graph_id[:split1], graph_id[split1:split2], graph_id[split2:]

    stats, train_x, val_x, test_x = normalization(train_x, val_x, test_x)

    base = os.path.basename(graph_signal_matrix_filename).split(".")[0]
    directory = os.path.dirname(graph_signal_matrix_filename)
    out = os.path.join(directory, f"{base}_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_astcgn.npz")

    np.savez_compressed(
        out,
        train_x=train_x.astype(np.float32),
        train_target=train_y.astype(np.float32),
        train_timestamp=train_ts,
        train_graph_id=train_gid,
        val_x=val_x.astype(np.float32),
        val_target=val_y.astype(np.float32),
        val_timestamp=val_ts,
        val_graph_id=val_gid,
        test_x=test_x.astype(np.float32),
        test_target=test_y.astype(np.float32),
        test_timestamp=test_ts,
        test_graph_id=test_gid,
        mean=stats["_mean"].astype(np.float32),
        std=stats["_std"].astype(np.float32),
        input_steps=np.asarray([input_steps], dtype=np.int32),
        pred_steps=np.asarray([num_for_predict], dtype=np.int32),
        alignment_trim=np.asarray([input_steps + num_for_predict - 1], dtype=np.int32),
    )

    print("Raw data:", data.shape)
    print("Input / output steps:", input_steps, num_for_predict)
    print("Alignment trim:", input_steps + num_for_predict - 1)
    print("train:", train_x.shape, train_y.shape, train_gid.shape)
    print("val  :", val_x.shape, val_y.shape, val_gid.shape)
    print("test :", test_x.shape, test_y.shape, test_gid.shape)
    print("mean :", stats["_mean"].shape)
    print("std  :", stats["_std"].shape)
    print("saved:", out)


if __name__ == "__main__":
    main()
