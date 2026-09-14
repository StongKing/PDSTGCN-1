import numpy as np
import torch


def _torch_mask(labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mean = mask.mean()
    if mean > 0:
        mask = mask / mean
    return torch.nan_to_num(mask)


def masked_mae(preds, labels, null_val=0.0):
    mask = _torch_mask(labels, null_val)
    return torch.mean(torch.nan_to_num(torch.abs(preds - labels) * mask))


def masked_mse(preds, labels, null_val=0.0):
    mask = _torch_mask(labels, null_val)
    return torch.mean(torch.nan_to_num((preds - labels) ** 2 * mask))


def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mape_np(y_true, y_pred, null_val=0.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = y_true != null_val
    denom = np.abs(y_true)
    valid = mask & (denom > 1e-8)
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(y_pred[valid] - y_true[valid]) / denom[valid]))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def rmse_np(y_true, y_pred):
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(d * d)))


def pdstgcn_loss(outputs, labels, alpha=0.01, beta=0.01):
    N = outputs.shape[1]

    loss_pred = torch.mean(
        torch.abs(outputs - labels)
    )

    loss_cons = torch.mean(
        torch.abs(
            outputs.sum(dim=1)
            - labels.sum(dim=1)
        )
    ) / N

    loss_int = torch.mean(
        torch.abs(
            outputs - torch.round(outputs)
        )
    )

    return (
        loss_pred
        + alpha * loss_cons
        + beta * loss_int
    )