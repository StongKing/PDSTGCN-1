from __future__ import annotations

# ============================================================
# OpenMP workaround for Windows / Anaconda
# Must be set before importing torch / numpy related libraries
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import configparser
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lib.metrics import masked_mae, masked_mse, pdstgcn_loss
from lib.utils import (
    get_adjacency_matrix,
    load_graphdata_channel1,
    predict_and_save_results_mstgcn,
)
from model.ASTGCN_r import make_model


# ============================================================
# USER SETTINGS
# ============================================================

# Two available modes:
#
# "train"
#     Train the model normally.
#
# "predict"
#     DO NOT train.
#     Load an already trained checkpoint and directly evaluate
#     it on the test set.
#
RUN_MODE = "predict"


# Used only when RUN_MODE == "predict"
#
# None:
#     Automatically load the best available checkpoint.
#
# Integer, e.g. 36:
#     Explicitly load epoch_36.params.
#
PREDICT_EPOCH = None


# Configuration file
CONFIG_PATH = "configurations/DIVVY_astgcn.conf"


# Physics-loss coefficients
#
# IMPORTANT:
# Training and validation use exactly the same coefficients.
#
LOSS_ALPHA = 0.005
LOSS_BETA = 0.01


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Validation loss
# ============================================================

def compute_validation_loss(
    net,
    val_loader,
    graph_store,
    device,
    alpha=LOSS_ALPHA,
    beta=LOSS_BETA,
):
    """
    Compute validation loss using exactly the same PDSTGCN
    objective as the training stage.

    No gradient update is performed.

    Validation:
        apply_rounding=False

    because the loss contains the integer penalty

        |x - round(x)|.

    Parameters
    ----------
    net
        PDSTGCN model.

    val_loader
        Validation DataLoader.

    graph_store
        Sparse dynamic graph store.

    device
        CUDA / CPU device.

    alpha
        Fleet-conservation coefficient.

    beta
        Integer-penalty coefficient.

    Returns
    -------
    float
        Mean validation loss.
    """

    net.eval()

    losses = []

    with torch.no_grad():

        for encoder_inputs, labels, graph_idx in val_loader:

            encoder_inputs = encoder_inputs.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # Dynamic graph corresponding exactly to this batch
            A = graph_store.dense(
                graph_idx.numpy(),
                device
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Validation must use continuous model output.
            # ------------------------------------------------
            outputs = net(
                encoder_inputs,
                A,
                apply_rounding=False
            )

            # ------------------------------------------------
            # Same loss as training
            # ------------------------------------------------
            loss = pdstgcn_loss(
                outputs,
                labels,
                alpha=alpha,
                beta=beta
            )

            losses.append(float(loss.item()))

    if not losses:
        return np.inf

    return float(np.mean(losses))


# ============================================================
# Find checkpoint for prediction
# ============================================================

def find_prediction_checkpoint(
    params_path: Path,
    predict_epoch=None
):
    """
    Find the checkpoint used for prediction.

    Priority
    --------
    1. If predict_epoch is specified:
           epoch_{predict_epoch}.params

    2. If best_model.params exists:
           best_model.params

    3. If best_epoch.txt exists:
           epoch_{best_epoch}.params

    4. Backward compatibility with old training runs:
           select the epoch_*.params file with the largest
           epoch number.

       In the current PDSTGCN training code, epoch checkpoints
       are saved only when validation loss reaches a new best.
       Therefore, for a completed old run, the largest saved
       epoch is the final best-validation checkpoint.
    """

    params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(
            f"\nCheckpoint directory does not exist:\n"
            f"{params_path}\n"
        )

    # ========================================================
    # Case 1: manually specified epoch
    # ========================================================

    if predict_epoch is not None:

        epoch = int(predict_epoch)

        ckpt = params_path / f"epoch_{epoch}.params"

        if not ckpt.exists():
            raise FileNotFoundError(
                f"\nSpecified checkpoint does not exist:\n"
                f"{ckpt}\n"
            )

        return ckpt, epoch


    # ========================================================
    # Case 2: fixed-name best model
    # ========================================================

    best_model_file = params_path / "best_model.params"

    if best_model_file.exists():

        epoch = None

        best_epoch_file = params_path / "best_epoch.txt"

        if best_epoch_file.exists():

            try:
                text = best_epoch_file.read_text(
                    encoding="utf-8"
                )

                for line in text.splitlines():

                    if line.startswith("best_epoch="):
                        epoch = int(
                            line.split("=", 1)[1].strip()
                        )
                        break

            except Exception:
                epoch = None

        return best_model_file, epoch


    # ========================================================
    # Case 3: best_epoch.txt exists
    # ========================================================

    best_epoch_file = params_path / "best_epoch.txt"

    if best_epoch_file.exists():

        text = best_epoch_file.read_text(
            encoding="utf-8"
        )

        epoch = None

        for line in text.splitlines():

            if line.startswith("best_epoch="):

                epoch = int(
                    line.split("=", 1)[1].strip()
                )

                break

        if epoch is not None:

            ckpt = params_path / f"epoch_{epoch}.params"

            if ckpt.exists():
                return ckpt, epoch


    # ========================================================
    # Case 4:
    # Old completed experiment
    #
    # Only improved-validation checkpoints were stored.
    # Therefore choose the largest saved epoch.
    # ========================================================

    files = list(
        params_path.glob("epoch_*.params")
    )

    if not files:
        raise FileNotFoundError(
            f"\nNo trained checkpoint was found in:\n"
            f"{params_path}\n"
        )

    def get_epoch_number(path: Path):
        return int(
            path.stem.split("_")[-1]
        )

    files = sorted(
        files,
        key=get_epoch_number
    )

    ckpt = files[-1]
    epoch = get_epoch_number(ckpt)

    return ckpt, epoch


# ============================================================
# Main
# ============================================================

def main(
    mode=RUN_MODE,
    predict_epoch=PREDICT_EPOCH
):

    # ========================================================
    # Project working directory
    # ========================================================

    project_root = Path(__file__).resolve().parent

    # Make all relative paths in the configuration file
    # relative to the PDSTGCN-1 project root.
    os.chdir(project_root)

    mode = str(mode).strip().lower()

    if mode not in {"train", "predict"}:
        raise ValueError(
            f'RUN_MODE must be "train" or "predict", '
            f"but got: {mode}"
        )


    # ========================================================
    # Read configuration
    # ========================================================

    config_path = project_root / CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n"
            f"{config_path}"
        )

    config = configparser.ConfigParser()

    config.read(
        config_path,
        encoding="utf-8"
    )

    if "Data" not in config or "Training" not in config:
        raise ValueError(
            f"Invalid configuration file:\n"
            f"{config_path}"
        )

    data_config = config["Data"]
    training_config = config["Training"]


    # ========================================================
    # Data settings
    # ========================================================

    adj_filename = data_config["adj_filename"]

    graph_signal_matrix_filename = (
        data_config["graph_signal_matrix_filename"]
    )

    dynamic_graph_filename = (
        data_config["dynamic_graph_filename"]
    )

    id_filename = data_config.get(
        "id_filename",
        fallback=None
    )

    num_of_vertices = int(
        data_config["num_of_vertices"]
    )

    points_per_hour = int(
        data_config["points_per_hour"]
    )

    num_for_predict = int(
        data_config["num_for_predict"]
    )

    len_input = int(
        data_config["len_input"]
    )

    dataset_name = data_config["dataset_name"]

    fleet_size = int(
        data_config["fleet_size"]
    )


    # ========================================================
    # Training settings
    # ========================================================

    ctx = training_config.get(
        "ctx",
        "0"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = ctx

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    learning_rate = float(
        training_config["learning_rate"]
    )

    epochs = int(
        training_config["epochs"]
    )

    start_epoch = int(
        training_config["start_epoch"]
    )

    batch_size = int(
        training_config["batch_size"]
    )

    num_of_weeks = int(
        training_config["num_of_weeks"]
    )

    num_of_days = int(
        training_config["num_of_days"]
    )

    num_of_hours = int(
        training_config["num_of_hours"]
    )

    time_strides = num_of_hours

    nb_chev_filter = int(
        training_config["nb_chev_filter"]
    )

    nb_time_filter = int(
        training_config["nb_time_filter"]
    )

    in_channels = int(
        training_config["in_channels"]
    )

    nb_block = int(
        training_config["nb_block"]
    )

    K = int(
        training_config["K"]
    )

    loss_function = (
        training_config["loss_function"]
        .lower()
    )

    metric_method = (
        training_config["metric_method"]
        .lower()
    )

    missing_value = float(
        training_config["missing_value"]
    )

    model_name = training_config["model_name"]

    seed = int(
        training_config.get(
            "seed",
            20260913
        )
    )

    set_seed(seed)


    # ========================================================
    # Basic consistency checks
    # ========================================================

    expected_input = (
        points_per_hour
        * num_of_hours
    )

    if expected_input != len_input:
        raise ValueError(
            f"len_input={len_input}, but "
            f"points_per_hour*num_of_hours="
            f"{expected_input}"
        )

    if len_input != 6 or num_for_predict != 6:

        print(
            f"WARNING: this Divvy project was designed "
            f"for 6->6; current config is "
            f"{len_input}->{num_for_predict}"
        )


    # ========================================================
    # Print experiment information
    # ========================================================

    print("=" * 90)

    if mode == "train":
        print("PDSTGCN TRAINING MODE")
    else:
        print("PDSTGCN PREDICTION-ONLY MODE")

    print("=" * 90)

    print(
        "Read configuration file:",
        config_path
    )

    print(
        "DEVICE:",
        device
    )

    print(
        "nodes / fleet:",
        num_of_vertices,
        fleet_size
    )

    print(
        "history / prediction:",
        len_input,
        num_for_predict
    )


    # ========================================================
    # Load prepared dataset
    # ========================================================

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


    # ========================================================
    # Static graph
    # ========================================================

    adj_mx, _ = get_adjacency_matrix(
        adj_filename,
        num_of_vertices,
        id_filename
    )


    # ========================================================
    # Build model
    # ========================================================

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


    # ========================================================
    # Checkpoint directory
    # ========================================================

    params_path = (
        Path("experiments")
        / dataset_name
        / (
            f"{model_name}_"
            f"h{num_of_hours}"
            f"d{num_of_days}"
            f"w{num_of_weeks}_"
            f"channel{in_channels}_"
            f"{learning_rate:.6e}"
        )
    )


    def checkpoint(epoch):
        return (
            params_path
            / f"epoch_{epoch}.params"
        )


    # ========================================================
    # MODE 1:
    # PREDICTION ONLY
    # ========================================================

    if mode == "predict":

        print("\n" + "=" * 90)
        print("PREDICTION ONLY")
        print("=" * 90)

        print(
            "Checkpoint directory:",
            params_path
        )

        # ----------------------------------------------------
        # Find checkpoint
        # ----------------------------------------------------

        ckpt, epoch = find_prediction_checkpoint(
            params_path=params_path,
            predict_epoch=predict_epoch
        )

        if epoch is None:
            print(
                "Selected checkpoint:",
                ckpt.name
            )
        else:
            print(
                f"Selected best epoch: {epoch}"
            )

            print(
                "Selected checkpoint:",
                ckpt.name
            )


        # ----------------------------------------------------
        # Load trained parameters
        # ----------------------------------------------------

        print(
            "Loading trained weights from:"
        )

        print(
            ckpt.resolve()
        )

        state_dict = torch.load(
            ckpt,
            map_location=device
        )

        net.load_state_dict(
            state_dict
        )


        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        # global_step is used only in the output filename.
        #
        # If best_model.params is loaded and its epoch number
        # is unavailable, use "best".
        #
        output_epoch = (
            epoch
            if epoch is not None
            else "best"
        )

        print("\n" + "-" * 90)

        print(
            "Running prediction on the TEST set..."
        )

        print("-" * 90)

        predict_and_save_results_mstgcn(
            net,
            test_loader,
            test_target_tensor,
            graph_store,
            output_epoch,
            str(params_path),
            "test",
            device
        )

        print("\n" + "=" * 90)

        print(
            "PREDICTION FINISHED"
        )

        print(
            "No training or parameter update was performed."
        )

        print("=" * 90)

        return


    # ========================================================
    # MODE 2:
    # TRAINING
    # ========================================================

    print("\n" + "=" * 90)

    print("TRAINING")

    print("=" * 90)


    # ========================================================
    # Criterion configuration
    #
    # The actual PDSTGCN train/validation loss below is
    # pdstgcn_loss().
    #
    # These definitions are retained for compatibility with
    # the existing configuration and project structure.
    # ========================================================

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

        raise ValueError(
            f"Unknown loss_function="
            f"{loss_function}"
        )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )


    # ========================================================
    # Prepare checkpoint directory
    # ========================================================

    if start_epoch == 0:

        if params_path.exists():
            shutil.rmtree(
                params_path
            )

        params_path.mkdir(
            parents=True,
            exist_ok=True
        )

    else:

        if not params_path.exists():
            raise FileNotFoundError(
                params_path
            )

        resume_ckpt = checkpoint(
            start_epoch
        )

        if not resume_ckpt.exists():
            raise FileNotFoundError(
                resume_ckpt
            )

        net.load_state_dict(
            torch.load(
                resume_ckpt,
                map_location=device
            )
        )


    # ========================================================
    # Print training settings
    # ========================================================

    print(
        "params_path:",
        params_path
    )

    print(
        "batch size:",
        batch_size,
        "K:",
        K,
        "filters:",
        nb_chev_filter,
        nb_time_filter
    )

    print(
        "config loss name:",
        loss_function
    )

    print(
        "metric:",
        metric_method
    )

    print(
        "PDSTGCN loss alpha / beta:",
        LOSS_ALPHA,
        LOSS_BETA
    )

    print(
        "IMPORTANT: zero inventory is a valid observation."
    )


    # ========================================================
    # Best model state
    # ========================================================

    best_val = np.inf
    best_epoch = None


    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        start_epoch,
        epochs
    ):

        # ----------------------------------------------------
        # 1. Validation of the CURRENT model
        #
        # This preserves the checkpoint timing convention used
        # by your current PDSTGCN project.
        # ----------------------------------------------------

        val_loss = compute_validation_loss(
            net,
            val_loader,
            graph_store,
            device,
            alpha=LOSS_ALPHA,
            beta=LOSS_BETA,
        )

        print(
            f"Epoch {epoch:03d} "
            f"pre-train val loss: "
            f"{val_loss:.6f}"
        )


        # ----------------------------------------------------
        # Save best validation checkpoint
        # ----------------------------------------------------

        if val_loss < best_val:

            best_val = val_loss
            best_epoch = epoch

            # Epoch-specific checkpoint
            torch.save(
                net.state_dict(),
                checkpoint(epoch)
            )

            # Fixed-name best checkpoint
            torch.save(
                net.state_dict(),
                params_path
                / "best_model.params"
            )

            # Best epoch information
            with open(
                params_path / "best_epoch.txt",
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"best_epoch="
                    f"{best_epoch}\n"
                )

                f.write(
                    f"best_val_loss="
                    f"{best_val:.12f}\n"
                )

                f.write(
                    f"alpha="
                    f"{LOSS_ALPHA}\n"
                )

                f.write(
                    f"beta="
                    f"{LOSS_BETA}\n"
                )

            print(
                f"New best validation model: "
                f"epoch {best_epoch}, "
                f"loss={best_val:.6f}"
            )


        # ----------------------------------------------------
        # 2. Training
        # ----------------------------------------------------

        net.train()

        train_losses = []

        for (
            encoder_inputs,
            labels,
            graph_idx
        ) in train_loader:

            encoder_inputs = (
                encoder_inputs.to(
                    device,
                    non_blocking=True
                )
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # Dynamic graph corresponding to the same samples
            A = graph_store.dense(
                graph_idx.numpy(),
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )


            # ------------------------------------------------
            # Continuous output during training
            # ------------------------------------------------

            outputs = net(
                encoder_inputs,
                A,
                apply_rounding=False
            )


            # ------------------------------------------------
            # PDSTGCN physics loss
            # ------------------------------------------------

            loss = pdstgcn_loss(
                outputs,
                labels,
                alpha=LOSS_ALPHA,
                beta=LOSS_BETA
            )


            # ------------------------------------------------
            # Back propagation
            # ------------------------------------------------

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                5.0
            )

            optimizer.step()

            train_losses.append(
                float(loss.item())
            )


        # ----------------------------------------------------
        # Epoch information
        # ----------------------------------------------------

        mean_train_loss = (
            float(
                np.mean(train_losses)
            )
            if train_losses
            else np.nan
        )

        print(
            f"Epoch {epoch:03d} "
            f"train loss: "
            f"{mean_train_loss:.6f} | "
            f"best val: "
            f"{best_val:.6f} "
            f"@ {best_epoch}"
        )


    # ========================================================
    # Training finished
    # ========================================================

    if best_epoch is None:
        raise RuntimeError(
            "Training finished without "
            "a valid checkpoint."
        )

    print("\n" + "=" * 90)

    print(
        "TRAINING FINISHED"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val:.6f}"
    )

    print("=" * 90)


    # ========================================================
    # Load the BEST model
    # ========================================================

    best_model_path = (
        params_path
        / "best_model.params"
    )

    net.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )


    # ========================================================
    # Final test prediction
    # ========================================================

    print(
        "\nRunning final test prediction "
        "using the best validation model..."
    )

    predict_and_save_results_mstgcn(
        net,
        test_loader,
        test_target_tensor,
        graph_store,
        best_epoch,
        str(params_path),
        "test",
        device
    )


# ============================================================
# Program entry
# ============================================================

if __name__ == "__main__":

    main(
        mode=RUN_MODE,
        predict_epoch=PREDICT_EPOCH
    )