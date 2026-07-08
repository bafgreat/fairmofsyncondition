import numpy as np
import torch

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sigmoid_logits(logits):
    if torch.is_tensor(logits):
        return torch.sigmoid(logits).detach().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-np.asarray(logits)))


# ============================================================
# MULTI-LABEL METRICS
# Per: solvent, metal_salt, modulator
# y_true: [N, C] multi-hot
# logits: [N, C]
# ============================================================

def multilabel_topk_hit_rate(logits, y_true, k=5):
    """
    Hit@k:
    vale 1 se almeno una label vera è nei top-k predetti.
    Molto utile per recommender di condizioni di sintesi.
    """
    scores = to_numpy(logits)
    y_true = to_numpy(y_true)

    topk = np.argsort(scores, axis=1)[:, -k:]

    hits = []
    for i in range(y_true.shape[0]):
        true_labels = set(np.where(y_true[i] > 0)[0])
        pred_labels = set(topk[i])

        if len(true_labels) == 0:
            continue

        hits.append(len(true_labels.intersection(pred_labels)) > 0)

    if len(hits) == 0:
        return np.nan

    return float(np.mean(hits))


def multilabel_precision_at_k(logits, y_true, k=5):
    scores = to_numpy(logits)
    y_true = to_numpy(y_true)

    topk = np.argsort(scores, axis=1)[:, -k:]

    values = []
    for i in range(y_true.shape[0]):
        true_labels = set(np.where(y_true[i] > 0)[0])
        pred_labels = set(topk[i])

        if len(true_labels) == 0:
            continue

        values.append(len(true_labels.intersection(pred_labels)) / k)

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def multilabel_recall_at_k(logits, y_true, k=5):
    scores = to_numpy(logits)
    y_true = to_numpy(y_true)

    topk = np.argsort(scores, axis=1)[:, -k:]

    values = []
    for i in range(y_true.shape[0]):
        true_labels = set(np.where(y_true[i] > 0)[0])
        pred_labels = set(topk[i])

        if len(true_labels) == 0:
            continue

        values.append(len(true_labels.intersection(pred_labels)) / len(true_labels))

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_average_precision(y_true, y_score, average="macro"):
    """
    PR-AUC / Average Precision.
    Per macro ignora classi senza positivi nel test set.
    """
    y_true = to_numpy(y_true)
    y_score = to_numpy(y_score)

    try:
        if average == "macro":
            valid_classes = y_true.sum(axis=0) > 0

            if valid_classes.sum() == 0:
                return np.nan

            return float(
                average_precision_score(
                    y_true[:, valid_classes],
                    y_score[:, valid_classes],
                    average="macro",
                )
            )

        return float(average_precision_score(y_true, y_score, average=average))

    except Exception:
        return np.nan


def compute_multilabel_metrics(
    logits,
    y_true,
    threshold=0.5,
    top_ks=(1, 3, 5, 10),
):
    """
    Usa questa per:
    - solvent
    - metal_salt
    - modulator
    """
    y_true = to_numpy(y_true).astype(int)
    y_score = sigmoid_logits(logits)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {}

    for k in top_ks:
        if k <= y_true.shape[1]:
            metrics[f"hit@{k}"] = multilabel_topk_hit_rate(y_score, y_true, k)
            metrics[f"precision@{k}"] = multilabel_precision_at_k(y_score, y_true, k)
            metrics[f"recall@{k}"] = multilabel_recall_at_k(y_score, y_true, k)

    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics["micro_precision"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)

    metrics["micro_recall"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    #metrics["micro_mAP"] = safe_average_precision(y_true, y_score, average="micro")
    #metrics["macro_mAP"] = safe_average_precision(y_true, y_score, average="macro")
    #metrics["weighted_mAP"] = safe_average_precision(y_true, y_score, average="weighted")

    return metrics


# ============================================================
# SINGLE-LABEL METRICS
# Per: synthetic_method, se usi CrossEntropyLoss
# y_true: [N] oppure one-hot [N, C]
# logits: [N, C]
# ============================================================

def get_class_indices(y):
    y = to_numpy(y)

    if y.ndim == 2:
        return np.argmax(y, axis=1)

    return y.astype(int)


def topk_accuracy(logits, y_true, k=3):
    scores = to_numpy(logits)
    y_true = get_class_indices(y_true)

    k = min(k, scores.shape[1])
    topk = np.argsort(scores, axis=1)[:, -k:]

    hits = [
        y_true[i] in topk[i]
        for i in range(len(y_true))
    ]

    return float(np.mean(hits))


def compute_singlelabel_metrics(logits, y_true, top_ks=(1, 2, 3, 5)):
    y_true = get_class_indices(y_true)
    scores = to_numpy(logits)
    y_pred = np.argmax(scores, axis=1)

    metrics = {}

    for k in top_ks:
        if k <= scores.shape[1]:
            metrics[f"top{k}_accuracy"] = topk_accuracy(scores, y_true, k)

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return metrics


# ============================================================
# REGRESSION METRICS
# Per: reaction_time, reaction_temperature
# y_true: [N, 1]
# y_pred: [N, 1]
# ============================================================

def compute_regression_metrics(y_pred, y_true):
    y_pred = to_numpy(y_pred).reshape(-1)
    y_true = to_numpy(y_true).reshape(-1)

    metrics = {}

    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["rmse"] = mean_squared_error(y_true, y_pred) ** 0.5
    metrics["median_ae"] = median_absolute_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)

    return metrics


def compute_temperature_metrics(y_pred, y_true):
    """
    Extra utile per reaction_temperature.
    """
    metrics = compute_regression_metrics(y_pred, y_true)

    y_pred = to_numpy(y_pred).reshape(-1)
    y_true = to_numpy(y_true).reshape(-1)

    abs_err = np.abs(y_pred - y_true)

    metrics["within_5C"] = float(np.mean(abs_err <= 5))
    metrics["within_10C"] = float(np.mean(abs_err <= 10))
    metrics["within_20C"] = float(np.mean(abs_err <= 20))

    return metrics


def compute_time_metrics(y_pred, y_true, eps=1e-8):
    """
    Extra utile per reaction_time.
    Misura anche se la predizione è entro fattore 2.
    """
    metrics = compute_regression_metrics(y_pred, y_true)

    y_pred = to_numpy(y_pred).reshape(-1)
    y_true = to_numpy(y_true).reshape(-1)

    y_pred_safe = np.maximum(y_pred, eps)
    y_true_safe = np.maximum(y_true, eps)

    ratio = np.maximum(
        y_pred_safe / y_true_safe,
        y_true_safe / y_pred_safe,
    )

    metrics["within_factor_2"] = float(np.mean(ratio <= 2.0))
    metrics["within_factor_3"] = float(np.mean(ratio <= 3.0))

    return metrics


# ============================================================
# PRETTY PRINT
# ============================================================

def print_metrics(metrics, prefix=""):
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        else:
            print(f"{prefix}{key}: {value}")

MULTILABEL_TARGETS = {
    "metal_salt",
    "solvent",
    "modulator",
}

SINGLELABEL_TARGETS = {
    "synthetic_method",
}

REGRESSION_TARGETS = {
    "reaction_time",
    "reaction_temperature",
}


def get_batch_target(batch, target_field):
    y = getattr(batch, target_field)
    y = y.float()
    y = y.reshape(batch.num_graphs, -1)
    return y


@torch.no_grad()
def evaluate_model_metrics(
    model,
    loader,
    device,
    target_field,
):
    """
    Valuta un modello su un loader e ritorna le metriche corrette
    in base al target_field.

    Input:
        model
        loader
        device
        target_field

    Output:
        metrics: dict
    """

    model.eval()
    model = model.to(device)

    all_outputs = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        outputs = model(batch)
        targets = get_batch_target(batch, target_field)

        all_outputs.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if target_field in MULTILABEL_TARGETS:
        metrics = compute_multilabel_metrics(
            logits=outputs,
            y_true=targets,
            top_ks=(1, 3, 5, 10),
        )

    elif target_field in SINGLELABEL_TARGETS:
        metrics = compute_singlelabel_metrics(
            logits=outputs,
            y_true=targets,
            top_ks=(1, 2, 3, 5),
        )

    elif target_field == "reaction_temperature":
        metrics = compute_temperature_metrics(
            y_pred=outputs,
            y_true=targets,
        )

    elif target_field == "reaction_time":
        metrics = compute_time_metrics(
            y_pred=outputs,
            y_true=targets,
        )

    else:
        raise ValueError(
            f"Target field sconosciuto: {target_field}. "
            "Aggiungilo a MULTILABEL_TARGETS, SINGLELABEL_TARGETS o REGRESSION_TARGETS."
        )
    return metrics