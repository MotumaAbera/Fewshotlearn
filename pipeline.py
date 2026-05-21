from typing import Dict, Optional

import torch

from dataset import create_datasets
from evaluate import evaluate_model
from loss import ContrastiveLoss
from model import SiameseNetwork
from report import generate_report, save_report
from train import train_model
from utils import ensure_directories, get_device, save_json, save_results_table, set_seed
from visualize import plot_loss_curves, plot_one_shot_examples, plot_sample_pairs, plot_tsne


def build_results_rows(metrics: Dict[str, object]):
    return [
        {"metric": "Dataset", "value": metrics["dataset"]},
        {"metric": "Best Validation Loss", "value": round(metrics["best_val_loss"], 4)},
        {"metric": "Final Train Pair Accuracy", "value": round(metrics["final_train_pair_accuracy"], 4)},
        {"metric": "Final Validation Pair Accuracy", "value": round(metrics["final_val_pair_accuracy"], 4)},
        {"metric": "One-Shot Accuracy", "value": round(metrics["one_shot_accuracy"], 4)},
        {"metric": "Embedding Dimension", "value": metrics["embedding_dim"]},
        {"metric": "Contrastive Margin", "value": metrics["margin"]},
    ]


def prepare_environment(config) -> torch.device:
    ensure_directories(
        [
            config.data_dir,
            config.outputs_dir,
            config.figures_dir,
            config.models_dir,
            config.results_dir,
            config.report_assets_dir,
        ]
    )
    set_seed(config.seed)
    return get_device()


def run_experiment(config, device: Optional[torch.device] = None) -> Dict[str, object]:
    device = device or prepare_environment(config)
    bundle = create_datasets(config)
    sample_batch = next(iter(bundle.train_loader))

    model = SiameseNetwork(embedding_dim=config.embedding_dim, input_channels=config.input_channels).to(device)
    criterion = ContrastiveLoss(margin=config.margin)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    plot_sample_pairs(
        batch=sample_batch,
        output_path=config.figures_dir / config.sample_pairs_name,
        report_path=config.report_assets_dir / config.sample_pairs_name,
    )

    history, best_metrics = train_model(
        config=config,
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
    plot_loss_curves(
        history=history,
        output_path=config.figures_dir / config.loss_curve_name,
        report_path=config.report_assets_dir / config.loss_curve_name,
    )

    model.load_state_dict(torch.load(config.models_dir / config.model_name, map_location=device))
    evaluation = evaluate_model(
        config=config,
        model=model,
        raw_test=bundle.raw_test,
        device=device,
        class_names=bundle.class_names,
    )
    plot_tsne(
        embeddings=evaluation["embeddings"],
        labels=evaluation["labels"],
        output_path=config.figures_dir / config.tsne_name,
        report_path=config.report_assets_dir / config.tsne_name,
    )
    plot_one_shot_examples(
        examples=evaluation["one_shot_examples"],
        raw_test=bundle.raw_test,
        output_path=config.figures_dir / config.one_shot_examples_name,
        report_path=config.report_assets_dir / config.one_shot_examples_name,
    )

    metrics = {
        "dataset": config.dataset_title,
        "device": str(device),
        "best_epoch": best_metrics["best_epoch"],
        "best_val_loss": best_metrics["best_val_loss"],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_pair_accuracy": history["train_pair_accuracy"][-1],
        "final_val_pair_accuracy": history["val_pair_accuracy"][-1],
        "one_shot_accuracy": evaluation["one_shot_accuracy"],
        "embedding_dim": config.embedding_dim,
        "margin": config.margin,
        "epochs_ran": len(history["train_loss"]),
        "one_shot_way": config.one_shot_way,
        "effective_one_shot_way": evaluation.get("effective_one_shot_way", config.one_shot_way),
        "one_shot_trials": config.one_shot_trials,
    }
    save_json(metrics, config.results_dir / config.metrics_name)
    save_json(evaluation["failure_analysis"], config.results_dir / f"failure_summary_{config.dataset_name}.json")
    rows = build_results_rows(metrics)
    save_results_table(rows, config.results_dir / config.results_table_name)
    save_results_table(rows, config.report_assets_dir / config.results_table_name)
    failure_rows = []
    for row in evaluation["failure_analysis"]["confusion_rows"]:
        failure_rows.append(
            {
                "analysis_type": "confusion",
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "count": row["count"],
                "trial": "",
                "query_index": "",
                "closest_distance": "",
            }
        )
    for row in evaluation["failure_analysis"]["hardest_rows"]:
        failure_rows.append(
            {
                "analysis_type": "hardest_failure",
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "count": "",
                "trial": row["trial"],
                "query_index": row["query_index"],
                "closest_distance": row["closest_distance"],
            }
        )
    if failure_rows:
        save_results_table(failure_rows, config.results_dir / config.failures_name)
    report_text = generate_report(config=config, metrics=metrics, history=history, evaluation=evaluation)
    save_report(report_text, config.project_root / config.report_name)
    save_report(report_text, config.report_assets_dir / config.report_name)

    return {
        "metrics": metrics,
        "history": history,
        "evaluation": evaluation,
        "device": device,
        "bundle": bundle,
    }
