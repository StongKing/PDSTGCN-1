# -*- coding: utf-8 -*-
"""
Offline reconciliation experiment for the current PDSTGCN checkpoint.

NO RETRAINING.

What this script does
---------------------
1. Load the already-trained best checkpoint.
2. Run the current model on validation and test sets.
3. Estimate node/horizon forecast uncertainty ONLY from validation residuals.
4. Apply post-hoc reconciliation to the test predictions:
      A. raw current prediction
      B. Euclidean simplex projection
      C. variance/MSE-weighted nonnegative reconciliation
   and optionally exact fleet-preserving integerization.
5. Compare station accuracy, node-0 accuracy, and fleet conservation.

The weighted reconciliation solves, independently for each sample and horizon,

    min_z  0.5 * sum_i (z_i - yhat_i)^2 / v_i
    s.t.   z_i >= 0,
           sum_i z_i = M,

where v_i is the validation error scale for node i at that horizon.
"""

from __future__ import annotations

# ============================================================
# OpenMP workaround for Windows / Anaconda
# MUST be set before importing numpy / scipy / torch
# ============================================================

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optional: avoid excessive OpenMP thread competition
os.environ.setdefault("OMP_NUM_THREADS", "1")

import configparser
from pathlib import Path

from lib.utils import (
    get_adjacency_matrix,
    load_graphdata_channel1,
)
from model.ASTGCN_r import make_model

import numpy as np
import pandas as pd
import torch


# =============================================================================
# USER SETTINGS
# =============================================================================

PROJECT_ROOT = Path(r"D:/PDSTGCN-1")

# Allow this script to be run from any working directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.utils import get_adjacency_matrix, load_graphdata_channel1
from model.ASTGCN_r import make_model

CONFIG_PATH = PROJECT_ROOT / "configurations" / "DIVVY_astgcn.conf"

CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "DIVVY"
    / "pdstgcn_h1d0w0_channel3_1.000000e-03"
    / "best_model.params"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "DIVVY"
    / "offline_reconciliation"
)

SHRINKAGE = 0.10
EPS = 1e-8
MODEL_APPLY_ROUNDING = True


# =============================================================================
# HELPERS
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(d * d)))


def mape_nonzero(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])))


# =============================================================================
# MODEL / DATA LOADING
# =============================================================================

def load_current_model_and_data():
    banner("LOAD CURRENT PDSTGCN CHECKPOINT + DATA")

    os.chdir(PROJECT_ROOT)

    require_file(CONFIG_PATH)
    require_file(CHECKPOINT)

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH, encoding="utf-8")

    data_config = config["Data"]
    train_config = config["Training"]

    adj_filename = data_config["adj_filename"]
    signal_filename = data_config["graph_signal_matrix_filename"]
    dynamic_filename = data_config["dynamic_graph_filename"]
    id_filename = data_config.get("id_filename", fallback=None)

    num_nodes = int(data_config["num_of_vertices"])
    fleet_size = int(data_config["fleet_size"])
    points_per_hour = int(data_config["points_per_hour"])
    num_for_predict = int(data_config["num_for_predict"])
    len_input = int(data_config["len_input"])

    num_hours = int(train_config["num_of_hours"])
    num_days = int(train_config["num_of_days"])
    num_weeks = int(train_config["num_of_weeks"])

    in_channels = int(train_config["in_channels"])
    nb_block = int(train_config["nb_block"])
    K = int(train_config["K"])
    nb_chev_filter = int(train_config["nb_chev_filter"])
    nb_time_filter = int(train_config["nb_time_filter"])

    batch_size = int(train_config.get("batch_size", 32))
    time_strides = num_hours

    ctx = train_config.get("ctx", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = ctx

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if points_per_hour * num_hours != len_input:
        raise RuntimeError(
            "Configuration mismatch: "
            f"points_per_hour*num_hours={points_per_hour*num_hours}, "
            f"len_input={len_input}"
        )

    (
        train_loader,
        train_target,
        val_loader,
        val_target,
        test_loader,
        test_target,
        graph_store,
        mean,
        std,
    ) = load_graphdata_channel1(
        signal_filename,
        dynamic_filename,
        num_hours,
        num_days,
        num_weeks,
        device,
        batch_size,
        in_channels,
        shuffle=False,
    )

    adj_mx, _ = get_adjacency_matrix(adj_filename, num_nodes, id_filename)

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
        num_nodes,
        fleet_size,
    )

    state = torch.load(CHECKPOINT, map_location=device)
    net.load_state_dict(state)
    net.eval()

    print("DEVICE                     :", device)
    print("Checkpoint                 :", CHECKPOINT)
    print("Nodes                      :", num_nodes)
    print("Fleet                      :", fleet_size)
    print("Validation samples         :", len(val_target))
    print("Test samples               :", len(test_target))
    print("Forecast horizons          :", num_for_predict)

    return net, val_loader, test_loader, graph_store, device, fleet_size


@torch.no_grad()
def predict_loader(net, loader, graph_store, device):
    pred = []
    target = []

    for x, y, graph_idx in loader:
        x = x.to(device, non_blocking=True)
        A = graph_store.dense(graph_idx.numpy(), device)

        yhat = net(
            x,
            A,
            apply_rounding=MODEL_APPLY_ROUNDING,
        )

        pred.append(yhat.detach().cpu().numpy())
        target.append(y.numpy())

    return (
        np.concatenate(pred, axis=0).astype(np.float64),
        np.concatenate(target, axis=0).astype(np.float64),
    )


# =============================================================================
# PROJECTION 1: ORDINARY EUCLIDEAN SIMPLEX
# =============================================================================

def euclidean_simplex_projection(pred, target_sum):
    pred = np.asarray(pred, dtype=np.float64)
    S, N, H = pred.shape

    y = pred.transpose(0, 2, 1).reshape(-1, N)
    u = np.sort(y, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - float(target_sum)
    ind = np.arange(1, N + 1, dtype=np.float64)[None, :]

    cond = u - cssv / ind > 0
    rho = np.maximum(cond.sum(axis=1) - 1, 0)

    theta = cssv[np.arange(y.shape[0]), rho] / (rho + 1.0)
    z = np.maximum(y - theta[:, None], 0.0)

    return z.reshape(S, H, N).transpose(0, 2, 1)


# =============================================================================
# PROJECTION 2: DIAGONAL UNCERTAINTY-WEIGHTED RECONCILIATION
# =============================================================================

def estimate_validation_error_scale(val_pred, val_target, shrinkage=0.10):
    err = np.asarray(val_pred, dtype=np.float64) - np.asarray(val_target, dtype=np.float64)

    # Mean squared residual by node and forecast horizon.
    v = np.mean(err * err, axis=0)  # [N,H]

    median_v = np.median(v, axis=0, keepdims=True)
    v = (1.0 - shrinkage) * v + shrinkage * median_v

    floor = np.maximum(median_v * 1e-6, EPS)
    v = np.maximum(v, floor)

    return v


def weighted_nonnegative_reconciliation(pred, error_scale, target_sum, n_bisect=70):
    pred = np.asarray(pred, dtype=np.float64)
    v_all = np.asarray(error_scale, dtype=np.float64)

    S, N, H = pred.shape

    if v_all.shape != (N, H):
        raise ValueError(f"error_scale shape {v_all.shape} != expected {(N,H)}")

    out = np.empty_like(pred)

    for h in range(H):
        y = pred[:, :, h]  # [S,N]
        v = np.maximum(v_all[:, h], EPS)  # [N]

        # KKT solution: z_i = max(y_i + lambda*v_i, 0).
        threshold = -y / v[None, :]

        lo = np.min(threshold, axis=1) - 1.0

        all_active_root = (float(target_sum) - np.sum(y, axis=1)) / np.sum(v)
        hi = np.maximum(
            np.max(threshold, axis=1) + 1.0,
            all_active_root + 1.0,
        )

        for _ in range(n_bisect):
            mid = (lo + hi) / 2.0
            z = np.maximum(y + mid[:, None] * v[None, :], 0.0)
            s = np.sum(z, axis=1)

            too_low = s < target_sum
            lo = np.where(too_low, mid, lo)
            hi = np.where(too_low, hi, mid)

        lam = (lo + hi) / 2.0
        z = np.maximum(y + lam[:, None] * v[None, :], 0.0)

        # Remove only floating-point residual.
        diff = float(target_sum) - np.sum(z, axis=1)
        j = int(np.argmax(v))
        z[:, j] += diff

        if np.min(z) < -1e-7:
            raise RuntimeError("Weighted reconciliation produced a negative value.")

        out[:, :, h] = np.maximum(z, 0.0)

    return out


# =============================================================================
# EXACT INTEGERIZATION
# =============================================================================

def integerize_preserve_sum(pred, target_sum):
    pred = np.asarray(pred, dtype=np.float64)
    S, N, H = pred.shape

    y = pred.transpose(0, 2, 1).reshape(-1, N)

    floor_y = np.floor(y)
    frac = y - floor_y
    remaining = int(target_sum) - floor_y.sum(axis=1).astype(np.int64)

    result = floor_y.copy()

    if np.any(remaining < 0):
        raise RuntimeError("Negative remaining fleet during integerization.")
    if np.any(remaining > N):
        raise RuntimeError("Remaining fleet exceeds node count.")

    for r_idx in range(result.shape[0]):
        k = int(remaining[r_idx])
        if k == 0:
            continue
        idx = np.argpartition(frac[r_idx], -k)[-k:]
        result[r_idx, idx] += 1.0

    return result.reshape(S, H, N).transpose(0, 2, 1)


# =============================================================================
# METRICS
# =============================================================================

def summarize_method(name, pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)

    p_station = pred[:, 1:, :]
    y_station = target[:, 1:, :]

    p0 = pred[:, 0, :]
    y0 = target[:, 0, :]

    fleet_pred = pred.sum(axis=1)
    fleet_true = target.sum(axis=1)

    fleet_error = fleet_pred - fleet_true
    fleet_abs = np.abs(fleet_error)

    return {
        "method": name,
        "station_MAE": mae(y_station, p_station),
        "station_RMSE": rmse(y_station, p_station),
        "station_MAPE": mape_nonzero(y_station, p_station),
        "node0_MAE": mae(y0, p0),
        "node0_RMSE": rmse(y0, p0),
        "allnode_RMSE": rmse(target, pred),
        "fleet_mean_abs": float(np.mean(fleet_abs)),
        "fleet_median_abs": float(np.median(fleet_abs)),
        "fleet_P95_abs": float(np.percentile(fleet_abs, 95)),
        "fleet_P99_abs": float(np.percentile(fleet_abs, 99)),
        "fleet_max_abs": float(np.max(fleet_abs)),
        "fleet_mean_signed": float(np.mean(fleet_error)),
        "fleet_exact_rate_pct": float(100.0 * np.mean(fleet_abs < 1e-6)),
    }


def adjustment_summary(name, reconciled, raw):
    d = np.asarray(reconciled) - np.asarray(raw)

    return {
        "method": name,
        "mean_abs_adjustment_node0": float(np.mean(np.abs(d[:, 0, :]))),
        "mean_abs_adjustment_stations": float(np.mean(np.abs(d[:, 1:, :]))),
        "max_abs_adjustment_node0": float(np.max(np.abs(d[:, 0, :]))),
        "max_abs_adjustment_stations": float(np.max(np.abs(d[:, 1:, :]))),
    }


def horizon_summary(method, pred, target):
    rows = []
    H = pred.shape[2]

    for h in range(H):
        pe = pred[:, :, h]
        yt = target[:, :, h]

        fleet_err = pe.sum(axis=1) - yt.sum(axis=1)

        rows.append(
            {
                "method": method,
                "horizon": h + 1,
                "station_MAE": mae(yt[:, 1:], pe[:, 1:]),
                "station_RMSE": rmse(yt[:, 1:], pe[:, 1:]),
                "node0_MAE": mae(yt[:, 0], pe[:, 0]),
                "fleet_mean_abs": float(np.mean(np.abs(fleet_err))),
                "fleet_max_abs": float(np.max(np.abs(fleet_err))),
            }
        )

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    net, val_loader, test_loader, graph_store, device, fleet_size = load_current_model_and_data()

    banner("RUN CURRENT CHECKPOINT ON VALIDATION + TEST")

    val_pred_raw, val_target = predict_loader(net, val_loader, graph_store, device)
    test_pred_raw, test_target = predict_loader(net, test_loader, graph_store, device)

    print("Validation prediction shape:", val_pred_raw.shape)
    print("Test prediction shape      :", test_pred_raw.shape)

    print(
        "Validation target max fleet error:",
        np.max(np.abs(val_target.sum(axis=1) - fleet_size)),
    )
    print(
        "Test target max fleet error      :",
        np.max(np.abs(test_target.sum(axis=1) - fleet_size)),
    )

    banner("ESTIMATE VALIDATION UNCERTAINTY")

    error_scale = estimate_validation_error_scale(
        val_pred_raw,
        val_target,
        shrinkage=SHRINKAGE,
    )

    node0_scale = error_scale[0, :]
    station_scale = error_scale[1:, :]

    print("Validation error scale - node0 by horizon:")
    print(np.array2string(node0_scale, precision=4))

    print("Median station error scale by horizon:")
    print(np.array2string(np.median(station_scale, axis=0), precision=4))

    print("Node0 / median-station error-scale ratio:")
    print(
        np.array2string(
            node0_scale / np.maximum(np.median(station_scale, axis=0), EPS),
            precision=3,
        )
    )

    banner("EUCLIDEAN SIMPLEX RECONCILIATION")

    test_euclidean_cont = euclidean_simplex_projection(test_pred_raw, fleet_size)
    test_euclidean_int = integerize_preserve_sum(test_euclidean_cont, fleet_size)

    banner("UNCERTAINTY-WEIGHTED RECONCILIATION")

    test_weighted_cont = weighted_nonnegative_reconciliation(
        test_pred_raw,
        error_scale,
        fleet_size,
    )
    test_weighted_int = integerize_preserve_sum(test_weighted_cont, fleet_size)

    banner("FINAL COMPARISON")

    methods = {
        "RAW_CURRENT": test_pred_raw,
        "EUCLIDEAN_CONT": test_euclidean_cont,
        "EUCLIDEAN_INT": test_euclidean_int,
        "WEIGHTED_CONT": test_weighted_cont,
        "WEIGHTED_INT": test_weighted_int,
    }

    summary_rows = []
    horizon_rows = []
    adjustment_rows = []

    for name, pred in methods.items():
        summary_rows.append(summarize_method(name, pred, test_target))
        horizon_rows.extend(horizon_summary(name, pred, test_target))

        if name != "RAW_CURRENT":
            adjustment_rows.append(adjustment_summary(name, pred, test_pred_raw))

    summary = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    adjustment_df = pd.DataFrame(adjustment_rows)

    columns = [
        "method",
        "station_MAE",
        "station_RMSE",
        "station_MAPE",
        "node0_MAE",
        "node0_RMSE",
        "allnode_RMSE",
        "fleet_mean_abs",
        "fleet_P95_abs",
        "fleet_max_abs",
        "fleet_mean_signed",
        "fleet_exact_rate_pct",
    ]

    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nAdjustment magnitude relative to RAW_CURRENT:")
    print(
        adjustment_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    summary_path = OUTPUT_DIR / "reconciliation_summary.csv"
    horizon_path = OUTPUT_DIR / "reconciliation_by_horizon.csv"
    adjustment_path = OUTPUT_DIR / "reconciliation_adjustment_summary.csv"
    prediction_path = OUTPUT_DIR / "reconciliation_predictions.npz"
    error_scale_path = OUTPUT_DIR / "validation_error_scale.npy"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    horizon_df.to_csv(horizon_path, index=False, encoding="utf-8-sig")
    adjustment_df.to_csv(adjustment_path, index=False, encoding="utf-8-sig")
    np.save(error_scale_path, error_scale)

    np.savez_compressed(
        prediction_path,
        target=test_target,
        raw=test_pred_raw,
        euclidean_cont=test_euclidean_cont,
        euclidean_int=test_euclidean_int,
        weighted_cont=test_weighted_cont,
        weighted_int=test_weighted_int,
        validation_error_scale=error_scale,
    )

    print("\nSaved:")
    print(summary_path)
    print(horizon_path)
    print(adjustment_path)
    print(prediction_path)
    print(error_scale_path)

    raw_station_mae = float(
        summary.loc[summary["method"] == "RAW_CURRENT", "station_MAE"].iloc[0]
    )
    weighted_station_mae = float(
        summary.loc[summary["method"] == "WEIGHTED_INT", "station_MAE"].iloc[0]
    )
    weighted_fleet_max = float(
        summary.loc[summary["method"] == "WEIGHTED_INT", "fleet_max_abs"].iloc[0]
    )

    banner("DECISION")

    print(f"RAW station MAE             : {raw_station_mae:.6f}")
    print(f"WEIGHTED_INT station MAE    : {weighted_station_mae:.6f}")
    print(f"WEIGHTED_INT max fleet error: {weighted_fleet_max:.6f}")
    print(f"Station MAE change          : {weighted_station_mae - raw_station_mae:+.6f}")

    if weighted_station_mae <= raw_station_mae and weighted_fleet_max < 1e-6:
        print(
            "RESULT: weighted reconciliation improves/equalizes station MAE "
            "while enforcing exact fleet conservation."
        )
    elif weighted_fleet_max < 1e-6:
        print(
            "RESULT: weighted reconciliation enforces exact fleet conservation. "
            "Check the station-MAE tradeoff before moving it into end-to-end training."
        )
    else:
        print(
            "WARNING: exact fleet conservation was not achieved; inspect numerical tolerances."
        )


if __name__ == "__main__":
    main()
