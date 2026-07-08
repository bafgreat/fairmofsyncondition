import os
import json
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import torch

from model.models import MOFConditionPredictor
from data.splits import make_train_val_test_split
from data.dataloaders import make_dataloaders
from data.data import get_data
from training.engine import fit
from training.metrics import evaluate_model_metrics


# ============================================================
# TARGET GROUPS
# ============================================================

MULTILABEL_TARGETS = {
    "solvent",
    "modulator",
    "metal_salt",
}

SINGLELABEL_TARGETS = {
    "synthetic_method",
}

REGRESSION_TARGETS = {
    "reaction_time",
    "reaction_temperature",
}

ALL_TARGETS = MULTILABEL_TARGETS | SINGLELABEL_TARGETS | REGRESSION_TARGETS


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_MIN_COUNT = {
    "solvent": 5,
    "modulator": 5,
    "metal_salt": 5,
    "synthetic_method": 55,
    "reaction_time": 0,
    "reaction_temperature": 0,
}

DEFAULT_MERGE_MIN_COUNT_CLASSES = {
    "solvent": False,
    "modulator": False,
    "metal_salt": False,
    "synthetic_method": True,
    "reaction_time": False,
    "reaction_temperature": False,
}

DEFAULT_METRIC_FNS = {
    "solvent": None,
    "modulator": None,
    "metal_salt": None,
    "synthetic_method": ["balanced_accuracy", "micro_f1", "macro_f1"],
    "reaction_time": None,
    "reaction_temperature": None,
}

DEFAULT_SELECTION_METRIC = {
    "solvent": "val_hit@1",
    "modulator": "val_hit@1",
    "metal_salt": "val_hit@1",
    "synthetic_method": "val_balanced_accuracy",
    "reaction_time": "best_val_loss",
    "reaction_temperature": "best_val_loss",
}


# ============================================================
# FORWARD ABLATION CONFIG
# ============================================================

BASE_GLOBAL_FIELDS = [
    "lattice",
]

CANDIDATE_GLOBAL_FIELDS = [
    "atomic_one_hot",
    "space_group_number",
    "crystal_system",
    "oms",
]


# ============================================================
# CLI UTILS
# ============================================================

def parse_csv_list(value):
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {"none", "null", ""}:
        return None

    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value):
    if value is None:
        return []

    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def resolve_metric_fns(metric_fns_arg, target_field):
    if metric_fns_arg == "auto":
        return DEFAULT_METRIC_FNS[target_field]

    return parse_csv_list(metric_fns_arg)


def resolve_selection_metric(selection_metric_arg, target_field):
    if selection_metric_arg == "auto":
        return DEFAULT_SELECTION_METRIC[target_field]

    return selection_metric_arg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forward structured ablation for MOF condition prediction."
    )

    parser.add_argument(
        "--lmdb-path",
        type=str,
        default="../../../final_mof_grap_for_training_data_v2.lmdb",
    )

    parser.add_argument(
        "--target-field",
        type=str,
        default="synthetic_method",
        choices=sorted(ALL_TARGETS),
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=None,
        help="If omitted, uses the default for the selected target.",
    )

    parser.add_argument(
        "--merge-min-count-classes",
        dest="merge_min_count_classes",
        action="store_true",
        help="Force merging low-count classes.",
    )

    parser.add_argument(
        "--no-merge-min-count-classes",
        dest="merge_min_count_classes",
        action="store_false",
        help="Force not merging low-count classes.",
    )

    parser.set_defaults(merge_min_count_classes=None)

    parser.add_argument(
        "--metric-fns",
        type=str,
        default="auto",
        help=(
            "Metrics passed to fit. Use 'auto', 'none', or comma-separated names. "
            "Example: balanced_accuracy,micro_f1,macro_f1"
        ),
    )

    parser.add_argument(
        "--selection-metric",
        type=str,
        default="auto",
        help=(
            "Metric used to select the best configuration. "
            "Use exact column name, e.g. val_balanced_accuracy, test_macro_f1, "
            "val_hit@1, test_hit@3, best_val_loss. "
            "Use 'auto' for target-dependent default."
        ),
    )

    parser.add_argument(
        "--selection-split",
        type=str,
        default="val",
        choices=["val", "test"],
        help=(
            "Used only if --selection-metric is passed without val_/test_ prefix. "
            "Example: --selection-metric balanced_accuracy --selection-split val "
            "selects val_balanced_accuracy."
        ),
    )

    parser.add_argument(
        "--selection-mode",
        type=str,
        default="auto",
        choices=["auto", "max", "min"],
        help="Whether the selection metric should be maximized or minimized.",
    )

    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help=(
            "If set, stops adding variables when no candidate improves over the current model. "
            "The full model is still evaluated at the end."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds. Example: 0,1,2",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--base-fields",
        type=str,
        default="lattice",
        help="Comma-separated base global fields. Default: lattice",
    )

    parser.add_argument(
        "--candidate-fields",
        type=str,
        default="atomic_one_hot,space_group_number,crystal_system,oms",
        help="Comma-separated candidate fields added one-by-one.",
    )

    return parser.parse_args()


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_target_dim(data_list, target_field):
    y = getattr(data_list[0], target_field)

    if target_field in REGRESSION_TARGETS:
        return 1

    return y.shape[-1]


def get_loss_fn(target_field):
    if target_field in MULTILABEL_TARGETS:
        return torch.nn.BCEWithLogitsLoss()

    if target_field in SINGLELABEL_TARGETS:
        # Nota: richiede target come class index nel train loop.
        # Se il tuo train loop usa one-hot, continua con BCEWithLogitsLoss.
        return torch.nn.BCEWithLogitsLoss()

    if target_field in REGRESSION_TARGETS:
        return torch.nn.SmoothL1Loss()

    raise ValueError(f"Target sconosciuto: {target_field}")

def make_summary(results_csv, summary_csv):
    df = pd.read_csv(results_csv)

    # ------------------------------------------------------------
    # Preserve forward-ablation order from the raw results CSV
    # ------------------------------------------------------------

    experiment_order = (
        df[["experiment"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "forward_order"})
    )

    # ------------------------------------------------------------
    # Metrics to summarize
    # ------------------------------------------------------------

    metric_cols = [
        c for c in df.columns
        if c.startswith("test_") and df[c].dtype != object
    ]

    # Optional but useful: include these losses too if present
    extra_metric_cols = [
        c for c in ["best_val_loss", "final_val_loss", "test_loss"]
        if c in df.columns and c not in metric_cols
    ]

    metric_cols = extra_metric_cols + metric_cols

    agg_dict = {c: ["mean", "std"] for c in metric_cols}

    # sort=False keeps the order of first appearance
    summary = df.groupby("experiment", sort=False).agg(agg_dict)

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    # ------------------------------------------------------------
    # Add order column and sort explicitly
    # ------------------------------------------------------------

    summary = summary.merge(
        experiment_order,
        on="experiment",
        how="left",
    )

    summary = summary.sort_values("forward_order").reset_index(drop=True)

    # Put forward_order as first column
    cols = ["forward_order", "experiment"] + [
        c for c in summary.columns
        if c not in {"forward_order", "experiment"}
    ]

    summary = summary[cols]

    summary.to_csv(summary_csv, index=False)

    return summary


def make_experiment_name(global_fields):
    if len(global_fields) == 0:
        return "graph_only"

    return "graph_" + "_".join(global_fields)


def deduplicate_keep_order(items):
    seen = set()
    out = []

    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)

    return out


def infer_selection_mode(metric_col, requested_mode):
    if requested_mode in {"max", "min"}:
        return requested_mode

    metric_lower = metric_col.lower()

    minimize_keywords = [
        "loss",
        "mae",
        "mse",
        "rmse",
        "error",
    ]

    if any(k in metric_lower for k in minimize_keywords):
        return "min"

    return "max"


def resolve_metric_column(df, selection_metric, selection_split):
    """
    Allows either exact column names:
        val_balanced_accuracy
        test_macro_f1
        best_val_loss

    Or short metric names:
        balanced_accuracy
        macro_f1
        hit@1

    In the second case, uses selection_split:
        val_balanced_accuracy or test_balanced_accuracy
    """

    if selection_metric in df.columns:
        return selection_metric

    candidates = []

    if selection_metric.startswith("val_") or selection_metric.startswith("test_"):
        candidates.append(selection_metric)
    else:
        candidates.append(f"{selection_split}_{selection_metric}")
        candidates.append(selection_metric)

        if selection_metric == "loss":
            if selection_split == "val":
                candidates.extend(["best_val_loss", "final_val_loss"])
            else:
                candidates.append("test_loss")

    for c in candidates:
        if c in df.columns:
            return c

    available = [
        c for c in df.columns
        if (
            c.startswith("val_")
            or c.startswith("test_")
            or c in {"best_val_loss", "final_val_loss", "test_loss"}
        )
    ]

    raise KeyError(
        "\nSelection metric non trovata.\n"
        f"Hai chiesto: {selection_metric}\n"
        f"Split: {selection_split}\n"
        f"Colonne disponibili:\n{available}\n"
    )


def aggregate_experiment_score(rows, experiment_name, metric_col):
    df = pd.DataFrame(rows)

    sub = df[df["experiment"] == experiment_name].copy()

    if sub.empty:
        raise ValueError(f"Nessun risultato trovato per experiment={experiment_name}")

    if metric_col not in sub.columns:
        raise KeyError(f"Colonna metrica non trovata: {metric_col}")

    values = pd.to_numeric(sub[metric_col], errors="coerce").dropna()

    if len(values) == 0:
        raise ValueError(
            f"La metrica {metric_col} per {experiment_name} contiene solo NaN."
        )

    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def is_better(candidate_score, current_score, mode):
    if current_score is None:
        return True

    if mode == "max":
        return candidate_score > current_score

    if mode == "min":
        return candidate_score < current_score

    raise ValueError(f"selection mode non valida: {mode}")


def save_selection_trace(trace_rows, selection_csv):
    if len(trace_rows) == 0:
        return

    pd.DataFrame(trace_rows).to_csv(selection_csv, index=False)


# ============================================================
# TRAINING RUNNER
# ============================================================

def run_experiment(
    experiment_name,
    global_fields,
    seeds,
    model_kwargs,
    train_loader,
    val_loader,
    test_loader,
    target_field,
    metric_fns,
    device,
    lr,
    weight_decay,
    epochs,
    patience,
    batch_size,
    results_csv,
    rows,
    evaluated_experiments,
):
    if experiment_name in evaluated_experiments:
        print(f"\nSkipping already evaluated experiment: {experiment_name}")
        return rows

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Experiment: {experiment_name}")
        print(f"Global fields: {global_fields}")
        print(f"Seed: {seed}")
        print("=" * 80)

        set_seed(seed)

        model = MOFConditionPredictor(
            target_dim=model_kwargs["target_dim"],
            global_fields=global_fields,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        loss_fn = get_loss_fn(target_field)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )

        model, history, test_loss = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            target_field=target_field,
            epochs=epochs,
            patience=patience,
            log_interval=1,
            metric_fns=metric_fns,
        )

        val_metrics = evaluate_model_metrics(
            model=model,
            loader=val_loader,
            device=device,
            target_field=target_field,
        )

        test_metrics = evaluate_model_metrics(
            model=model,
            loader=test_loader,
            device=device,
            target_field=target_field,
        )

        row = {
            "experiment": experiment_name,
            "target_field": target_field,
            "seed": seed,
            "global_fields": json.dumps(global_fields),
            "target_dim": model_kwargs["target_dim"],
            "num_parameters": count_parameters(model),
            "best_val_loss": min(history["val_loss"]),
            "final_val_loss": history["val_loss"][-1],
            "test_loss": test_loss,
            "epochs_trained": len(history["val_loss"]),
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "patience": patience,
        }

        for k, v in val_metrics.items():
            row[f"val_{k}"] = v

        for k, v in test_metrics.items():
            row[f"test_{k}"] = v

        rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(results_csv, index=False)

        print("\nCurrent result:")
        print(f"experiment: {experiment_name}")
        print(f"seed:       {seed}")

        if target_field in MULTILABEL_TARGETS:
            print(
                "test hit@1: {0:.4f} | hit@3: {1:.4f} | hit@5: {2:.4f}".format(
                    test_metrics.get("hit@1", np.nan),
                    test_metrics.get("hit@3", np.nan),
                    test_metrics.get("hit@5", np.nan),
                )
            )

        if target_field in SINGLELABEL_TARGETS:
            print(
                "test balanced_accuracy: {0:.4f} | micro_f1: {1:.4f} | macro_f1: {2:.4f}".format(
                    test_metrics.get("balanced_accuracy", np.nan),
                    test_metrics.get("micro_f1", np.nan),
                    test_metrics.get("macro_f1", np.nan),
                )
            )

        print(f"Saved partial results to: {results_csv}")

    evaluated_experiments.add(experiment_name)

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    warnings.filterwarnings(
        "ignore",
        message="The usage of `scatter\\(reduce='max'\\).*",
        category=UserWarning,
    )

    target_field = args.target_field

    min_count = (
        args.min_count
        if args.min_count is not None
        else DEFAULT_MIN_COUNT[target_field]
    )

    merge_min_count_classes = (
        args.merge_min_count_classes
        if args.merge_min_count_classes is not None
        else DEFAULT_MERGE_MIN_COUNT_CLASSES[target_field]
    )

    metric_fns = resolve_metric_fns(args.metric_fns, target_field)

    selection_metric = resolve_selection_metric(
        args.selection_metric,
        target_field,
    )

    seeds = parse_int_list(args.seeds)

    base_fields = deduplicate_keep_order(parse_csv_list(args.base_fields))
    candidate_fields = deduplicate_keep_order(parse_csv_list(args.candidate_fields))

    candidate_fields = [
        f for f in candidate_fields
        if f not in base_fields
    ]

    full_fields = deduplicate_keep_order(base_fields + candidate_fields)

    os.makedirs(args.results_dir, exist_ok=True)

    results_csv = os.path.join(
        args.results_dir,
        f"{target_field}_forward_ablation_results.csv",
    )

    summary_csv = os.path.join(
        args.results_dir,
        f"{target_field}_forward_ablation_summary.csv",
    )

    split_path = os.path.join(
        args.results_dir,
        f"{target_field}_split_indices.npz",
    )

    selection_csv = os.path.join(
        args.results_dir,
        f"{target_field}_forward_selection_trace.csv",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Target: {target_field}")
    print(f"Min count: {min_count}")
    print(f"Merge min-count classes: {merge_min_count_classes}")
    print(f"Metric fns: {metric_fns}")
    print(f"Selection metric: {selection_metric}")
    print(f"Selection split: {args.selection_split}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Base fields: {base_fields}")
    print(f"Candidate fields: {candidate_fields}")
    print(f"Full fields: {full_fields}")
    print(f"Seeds: {seeds}")

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    data_list = get_data(
        lmdb_path=args.lmdb_path,
        field=target_field,
        min_count=min_count,
        merge_min_count_classess=merge_min_count_classes,
    )

    print(len(data_list))
    target_dim = get_target_dim(data_list, target_field)

    print(f"Dataset size: {len(data_list)}")
    print(f"Target dim: {target_dim}")

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------

    train_data, val_data, test_data, train_idx, val_idx, test_idx = make_train_val_test_split(
        dataset=data_list,
        target_field=target_field,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=args.split_seed,
    )

    np.savez(
        split_path,
        train_idx=np.array(train_idx),
        val_idx=np.array(val_idx),
        test_idx=np.array(test_idx),
    )

    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")
    print(f"Saved split to: {split_path}")

    train_loader, val_loader, test_loader = make_dataloaders(
        train_data,
        val_data,
        test_data,
        batch_size=args.batch_size,
    )

    model_kwargs = {
        "target_dim": target_dim,
    }

    rows = []
    trace_rows = []
    evaluated_experiments = set()

    # ------------------------------------------------------------
    # 1) Always start with graph + lattice
    # ------------------------------------------------------------

    current_fields = list(base_fields)
    current_experiment = make_experiment_name(current_fields)

    rows = run_experiment(
        experiment_name=current_experiment,
        global_fields=current_fields,
        seeds=seeds,
        model_kwargs=model_kwargs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_field=target_field,
        metric_fns=metric_fns,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        results_csv=results_csv,
        rows=rows,
        evaluated_experiments=evaluated_experiments,
    )

    df_tmp = pd.DataFrame(rows)
    metric_col = resolve_metric_column(
        df=df_tmp,
        selection_metric=selection_metric,
        selection_split=args.selection_split,
    )

    selection_mode = infer_selection_mode(
        metric_col=metric_col,
        requested_mode=args.selection_mode,
    )

    current_score, current_std = aggregate_experiment_score(
        rows=rows,
        experiment_name=current_experiment,
        metric_col=metric_col,
    )

    print("\n" + "-" * 80)
    print("Initial model")
    print(f"Experiment: {current_experiment}")
    print(f"Fields: {current_fields}")
    print(f"Selection metric column: {metric_col}")
    print(f"Selection mode: {selection_mode}")
    print(f"Score mean ± std: {current_score:.6f} ± {current_std:.6f}")
    print("-" * 80)

    # ------------------------------------------------------------
    # 2) Forward selection
    # ------------------------------------------------------------

    remaining_fields = list(candidate_fields)
    round_id = 1

    while len(remaining_fields) > 0:
        print("\n" + "#" * 80)
        print(f"FORWARD ROUND {round_id}")
        print(f"Current fields: {current_fields}")
        print(f"Remaining fields: {remaining_fields}")
        print("#" * 80)

        candidate_configs = []

        for new_field in remaining_fields:
            fields = current_fields + [new_field]
            experiment_name = make_experiment_name(fields)

            candidate_configs.append(
                {
                    "new_field": new_field,
                    "fields": fields,
                    "experiment_name": experiment_name,
                }
            )

            rows = run_experiment(
                experiment_name=experiment_name,
                global_fields=fields,
                seeds=seeds,
                model_kwargs=model_kwargs,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                target_field=target_field,
                metric_fns=metric_fns,
                device=device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                results_csv=results_csv,
                rows=rows,
                evaluated_experiments=evaluated_experiments,
            )

        df_tmp = pd.DataFrame(rows)
        metric_col = resolve_metric_column(
            df=df_tmp,
            selection_metric=selection_metric,
            selection_split=args.selection_split,
        )

        selection_mode = infer_selection_mode(
            metric_col=metric_col,
            requested_mode=args.selection_mode,
        )

        ranking = []

        for cfg in candidate_configs:
            score_mean, score_std = aggregate_experiment_score(
                rows=rows,
                experiment_name=cfg["experiment_name"],
                metric_col=metric_col,
            )

            ranking.append(
                {
                    "round": round_id,
                    "current_before": json.dumps(current_fields),
                    "remaining_before": json.dumps(remaining_fields),
                    "new_field": cfg["new_field"],
                    "experiment": cfg["experiment_name"],
                    "fields": cfg["fields"],
                    "score_mean": score_mean,
                    "score_std": score_std,
                }
            )

        reverse = selection_mode == "max"
        ranking = sorted(
            ranking,
            key=lambda x: x["score_mean"],
            reverse=reverse,
        )

        best = ranking[0]

        print("\nCandidate ranking:")
        for i, item in enumerate(ranking, start=1):
            print(
                f"{i}. {item['experiment']} | "
                f"{metric_col}: {item['score_mean']:.6f} ± {item['score_std']:.6f}"
            )

        should_accept = True

        if args.require_improvement:
            should_accept = is_better(
                candidate_score=best["score_mean"],
                current_score=current_score,
                mode=selection_mode,
            )

        for item in ranking:
            item["metric_col"] = metric_col
            item["selection_mode"] = selection_mode
            item["current_score_before"] = current_score
            item["selected"] = bool(item["experiment"] == best["experiment"] and should_accept)
            trace_rows.append(item)

        save_selection_trace(trace_rows, selection_csv)

        if not should_accept:
            print("\nNo candidate improved over the current model.")
            print("Stopping forward selection because --require-improvement was set.")
            break

        selected_new_field = best["new_field"]
        current_fields = list(best["fields"])
        current_experiment = best["experiment"]
        current_score = best["score_mean"]
        current_std = best["score_std"]

        remaining_fields = [
            f for f in remaining_fields
            if f != selected_new_field
        ]

        print("\nSelected for next round:")
        print(f"Experiment: {current_experiment}")
        print(f"Added field: {selected_new_field}")
        print(f"Fields: {current_fields}")
        print(f"{metric_col}: {current_score:.6f} ± {current_std:.6f}")
        print(f"Saved selection trace to: {selection_csv}")

        round_id += 1

    # ------------------------------------------------------------
    # 3) Ensure full model is evaluated
    # ------------------------------------------------------------

    full_experiment = make_experiment_name(full_fields)

    if full_experiment not in evaluated_experiments:
        print("\n" + "#" * 80)
        print("FINAL FULL MODEL")
        print(f"Full fields: {full_fields}")
        print("#" * 80)

        rows = run_experiment(
            experiment_name=full_experiment,
            global_fields=full_fields,
            seeds=seeds,
            model_kwargs=model_kwargs,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            target_field=target_field,
            metric_fns=metric_fns,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            results_csv=results_csv,
            rows=rows,
            evaluated_experiments=evaluated_experiments,
        )

    # ------------------------------------------------------------
    # Summary paper-ready
    # ------------------------------------------------------------

    summary = make_summary(
        results_csv=results_csv,
        summary_csv=summary_csv,
    )

    print("\nFinished.")
    print(f"Saved results to: {results_csv}")
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved selection trace to: {selection_csv}")
    print(summary)


if __name__ == "__main__":
    main()