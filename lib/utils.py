from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.utils.data
from scipy.sparse.linalg import eigs
from lib.metrics import pdstgcn_loss

from .metrics import mae_np, rmse_np, masked_mape_np


def re_normalization(x, mean, std):
    return x * std + mean


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if "npy" in str(distance_df_filename):
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    D = np.zeros_like(A)
    id_dict = None
    if id_filename:
        with open(id_filename, "r", encoding="utf-8") as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))}

    with open(distance_df_filename, "r", encoding="utf-8") as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if id_dict is not None:
                i, j = id_dict[i], id_dict[j]
            A[i, j] = 1.0
            D[i, j] = distance
    return A, D


def first_order_graph(W):
    W = np.asarray(W, dtype=np.float32)
    Wt = W + np.eye(W.shape[0], dtype=np.float32)
    deg = Wt.sum(axis=1)
    inv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    return (inv[:, None] * Wt * inv[None, :]).astype(np.float32)


def scaled_Laplacian(W):
    W = np.asarray(W, dtype=np.float64)
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    if np.allclose(L, 0):
        return -np.eye(W.shape[0], dtype=np.float32)
    try:
        lam = float(eigs(L, k=1, which="LR", return_eigenvectors=False)[0].real)
    except Exception:
        lam = float(np.max(np.linalg.eigvals(L).real))
    lam = max(lam, 1e-8)
    return ((2.0 * L) / lam - np.eye(W.shape[0])).astype(np.float32)


def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    if K <= 0:
        return []
    if K == 1:
        return [np.eye(N, dtype=np.float32)]
    polys = [np.eye(N, dtype=np.float32), L_tilde.copy().astype(np.float32)]
    for _ in range(2, K):
        polys.append((2 * L_tilde @ polys[-1] - polys[-2]).astype(np.float32))
    return polys


@dataclass
class SparseDynamicGraphStore:
    graph_ptr: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    num_nodes: int
    num_graphs: int

    @classmethod
    def load(cls, filename: str | os.PathLike):
        z = np.load(filename, allow_pickle=False)
        required = {"graph_ptr", "src", "dst", "weight", "num_nodes", "num_graphs"}
        missing = required - set(z.files)
        if missing:
            raise KeyError(f"Dynamic graph file missing keys: {sorted(missing)}")
        return cls(
            graph_ptr=z["graph_ptr"].astype(np.int64),
            src=z["src"].astype(np.int64),
            dst=z["dst"].astype(np.int64),
            weight=z["weight"].astype(np.float32),
            num_nodes=int(z["num_nodes"].reshape(-1)[0]),
            num_graphs=int(z["num_graphs"].reshape(-1)[0]),
        )

    def dense(self, graph_ids, device):
        ids = np.asarray(graph_ids, dtype=np.int64).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.num_graphs):
            raise IndexError(f"graph id out of range [0,{self.num_graphs}): {ids}")
        out = torch.zeros((len(ids), self.num_nodes, self.num_nodes), dtype=torch.float32, device=device)
        for b, gid in enumerate(ids):
            a, c = int(self.graph_ptr[gid]), int(self.graph_ptr[gid + 1])
            if c <= a:
                continue
            src = torch.as_tensor(self.src[a:c], dtype=torch.long, device=device)
            dst = torch.as_tensor(self.dst[a:c], dtype=torch.long, device=device)
            val = torch.as_tensor(self.weight[a:c], dtype=torch.float32, device=device)
            out[b].index_put_((src, dst), val, accumulate=True)
        return out


def _make_loader(x, y, graph_id, device, batch_size, shuffle):
    xt = torch.from_numpy(x).float()  # keep on CPU; move per batch to avoid GPU OOM
    yt = torch.from_numpy(y).float()
    gt = torch.from_numpy(graph_id.reshape(-1)).long()
    dataset = torch.utils.data.TensorDataset(xt, yt, gt)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return loader, yt


def load_graphdata_channel1(
    graph_signal_matrix_filename,
    dynamic_graph_filename,
    num_of_hours,
    num_of_days,
    num_of_weeks,
    DEVICE,
    batch_size,
    in_channels,
    shuffle=True,
):
    file = os.path.basename(graph_signal_matrix_filename).split(".")[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)
    filename = os.path.join(
        dirpath,
        file + f"_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_astcgn.npz",
    )
    print("load prepared samples:", filename)
    f = np.load(filename, allow_pickle=False)

    def take(prefix):
        x = f[f"{prefix}_x"][:, :, 0:in_channels, :].astype(np.float32)
        y = f[f"{prefix}_target"].astype(np.float32)
        gid_key = f"{prefix}_graph_id"
        if gid_key in f.files:
            gid = f[gid_key].astype(np.int64)
        else:
            # Backward compatibility: label timestamp - input length.
            ts = f[f"{prefix}_timestamp"].astype(np.int64)
            gid = ts - x.shape[-1]
        return x, y, gid

    train_x, train_y, train_gid = take("train")
    val_x, val_y, val_gid = take("val")
    test_x, test_y, test_gid = take("test")
    mean = f["mean"][:, :, 0:in_channels, :]
    std = f["std"][:, :, 0:in_channels, :]

    graph_store = SparseDynamicGraphStore.load(dynamic_graph_filename)
    total_samples = train_x.shape[0] + val_x.shape[0] + test_x.shape[0]
    if graph_store.num_graphs != total_samples:
        raise AssertionError(
            f"Dynamic graph count {graph_store.num_graphs} != prepared sample count {total_samples}. "
            "Run data_generate.py and prepareData.py from the same Divvy dataset."
        )
    if graph_store.num_nodes != train_x.shape[1]:
        raise AssertionError(
            f"Dynamic graph nodes {graph_store.num_nodes} != signal nodes {train_x.shape[1]}"
        )

    train_loader, train_target_tensor = _make_loader(train_x, train_y, train_gid, DEVICE, batch_size, shuffle)
    val_loader, val_target_tensor = _make_loader(val_x, val_y, val_gid, DEVICE, batch_size, False)
    test_loader, test_target_tensor = _make_loader(test_x, test_y, test_gid, DEVICE, batch_size, False)

    print("train:", train_x.shape, train_y.shape)
    print("val  :", val_x.shape, val_y.shape)
    print("test :", test_x.shape, test_y.shape)
    print("dynamic graphs:", graph_store.num_graphs, "nodes:", graph_store.num_nodes)

    return (
        train_loader,
        train_target_tensor,
        val_loader,
        val_target_tensor,
        test_loader,
        test_target_tensor,
        graph_store,
        mean,
        std,
    )


def compute_val_loss_mstgcn(net, val_loader, graph_store, criterion, masked_flag, missing_value, device):
    net.eval()
    losses = []
    with torch.no_grad():
        for encoder_inputs, labels, graph_idx in val_loader:
            encoder_inputs = encoder_inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            A = graph_store.dense(graph_idx.numpy(), device)
            outputs = net(encoder_inputs, A, apply_rounding=False)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                # loss = criterion(outputs, labels)
                loss = pdstgcn_loss(outputs, labels)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else np.inf


def predict_and_save_results_mstgcn(
    net,
    data_loader,
    data_target_tensor,
    graph_store,
    global_step,
    params_path,
    type_name,
    device,
):
    net.eval()
    pred, target, inputs = [], [], []
    with torch.no_grad():
        for encoder_inputs, labels, graph_idx in data_loader:
            x = encoder_inputs.to(device, non_blocking=True)
            A = graph_store.dense(graph_idx.numpy(), device)
            yhat = net(x, A, apply_rounding=True)
            pred.append(yhat.cpu().numpy())
            target.append(labels.numpy())
            inputs.append(encoder_inputs[:, :, 0:1, :].numpy())
    pred = np.concatenate(pred, axis=0)
    target = np.concatenate(target, axis=0)
    inputs = np.concatenate(inputs, axis=0)

    os.makedirs(params_path, exist_ok=True)
    out = os.path.join(params_path, f"output_epoch_{global_step}_{type_name}.npz")
    np.savez_compressed(out, input=inputs, prediction=pred, data_target_tensor=target)

    # ============================================================
    # Evaluation metrics: physical stations only
    # Node 0 is the mesoscopic in-transit node and is NOT included
    # in MAE / RMSE / MAPE.
    # ============================================================

    target_eval = target[:, 1:, :]
    pred_eval = pred[:, 1:, :]

    mae = mae_np(target_eval, pred_eval)
    rmse = rmse_np(target_eval, pred_eval)
    mape = masked_mape_np(target_eval, pred_eval, 0.0)

    print(f"{type_name} MAE : {mae:.6f}")
    print(f"{type_name} RMSE: {rmse:.6f}")
    print(f"{type_name} MAPE(nonzero only): {mape:.6f}")

    fleet_true = target.sum(axis=1)
    fleet_pred = pred.sum(axis=1)
    print(f"{type_name} max |fleet error|: {np.max(np.abs(fleet_pred - fleet_true)):.6f}")
    print("saved:", out)
    return pred
