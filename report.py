from pathlib import Path
from typing import Dict, List


def generate_report(config, metrics: Dict[str, object], history: Dict[str, List[float]], evaluation: Dict[str, object]) -> str:
    confusion_rows = evaluation["failure_analysis"]["confusion_rows"][:5]
    confusion_lines = "\n".join(
        [f"- {row['true_label']} -> {row['predicted_label']}: {row['count']} cases" for row in confusion_rows]
    )
    if not confusion_lines:
        confusion_lines = "- No one-shot failures were recorded in the current run."

    return f"""# Few-Shot Learning with Siamese Networks Report

## Project Summary

This experiment trains a Siamese Neural Network with contrastive loss on {config.dataset_title}. The model learns an embedding space where same-class images stay close together while different-class images are pushed apart. The learned embedding is then evaluated through one-shot recognition rather than conventional closed-set classification.

## Experimental Setup

- Dataset: {config.dataset_title}
- Epochs run: {metrics['epochs_ran']}
- Batch size: {config.batch_size}
- Train pairs per epoch: {config.train_pairs_per_epoch}
- Validation pairs: {config.val_pairs}
- Embedding dimension: {config.embedding_dim}
- Contrastive margin: {config.margin}
- One-shot setting: {config.one_shot_way}-way, {config.one_shot_trials} trials

## Quantitative Results

- Best validation loss: {metrics['best_val_loss']:.4f}
- Final train pair accuracy: {metrics['final_train_pair_accuracy']:.4f}
- Final validation pair accuracy: {metrics['final_val_pair_accuracy']:.4f}
- One-shot accuracy: {metrics['one_shot_accuracy']:.4f}

## Learning Dynamics

The training curve is saved in `report_assets/{config.loss_curve_name}`. A healthy run should show decreasing training loss together with stable or improving validation loss. If validation loss stops improving quickly, that usually indicates either a small model capacity ceiling or that the sampled negative pairs are too easy.

## Embedding Space Analysis

The t-SNE projection saved in `report_assets/{config.tsne_name}` provides a low-dimensional view of the learned embedding space. Effective metric learning should create visible clusters for individual classes, while visually similar classes remain closer to each other. This behavior is expected because t-SNE preserves local neighborhood structure more than global geometry.

## One-Shot Failure Analysis

The model made {evaluation['failure_analysis']['failure_count']} errors out of {evaluation['failure_analysis']['total_trials']} one-shot trials.

Most common confusion patterns:
{confusion_lines}

Hardest failure cases are saved in `outputs/results/{config.failures_name}`.

## Discussion

Few-shot learning is useful when annotated data is limited, expensive, or constantly changing. Instead of learning only a fixed classifier boundary, the Siamese network learns a reusable similarity function. That is especially useful for verification, retrieval, anomaly matching, and low-resource recognition tasks.

For this project, {config.dataset_title} offers a fast and controlled benchmark. MNIST-style datasets let the model converge quickly on CPU, which makes them practical for university coursework and rapid experimentation. The main limitation is that these datasets are relatively simple compared with real-world image variation, so they can overestimate how well a method will scale without harder negative mining or stronger feature extractors.

## Reproducibility

Run the full experiment with:

```bash
python main.py --dataset {config.dataset_name}
```

Launch the frontend with:

```bash
streamlit run app.py
```
"""


def save_report(report_text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
