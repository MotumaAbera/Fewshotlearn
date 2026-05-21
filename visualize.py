from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils import copy_file


def _save_dual(path_a: Path, path_b: Path) -> None:
    copy_file(path_a, path_b)


def _to_display_image(tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    return image


def plot_loss_curves(history: Dict[str, List[float]], output_path: Path, report_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("Training Dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    _save_dual(output_path, report_path)


def plot_sample_pairs(batch, output_path: Path, report_path: Path, max_pairs: int = 6) -> None:
    left, right, pair_label, left_digit, right_digit = batch
    rows = min(max_pairs, left.size(0))
    fig, axes = plt.subplots(rows, 2, figsize=(4, 2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(rows):
        left_image = _to_display_image(left[row])
        right_image = _to_display_image(right[row])
        axes[row, 0].imshow(left_image, cmap="gray" if left_image.ndim == 2 else None)
        axes[row, 0].set_title(f"Anchor: {int(left_digit[row])}")
        axes[row, 1].imshow(right_image, cmap="gray" if right_image.ndim == 2 else None)
        relation = "Negative" if int(pair_label[row].item()) == 1 else "Positive"
        axes[row, 1].set_title(f"Pair: {int(right_digit[row])} ({relation})")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    _save_dual(output_path, report_path)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, report_path: Path) -> None:
    perplexity = max(2, min(30, len(embeddings) - 1))
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
    projection = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", s=12, alpha=0.8)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Projection of Siamese Embeddings")
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    _save_dual(output_path, report_path)


def plot_one_shot_examples(examples: List[Dict[str, object]], raw_test, output_path: Path, report_path: Path) -> None:
    rows = len(examples)
    cols = 1 + len(examples[0]["support_indices"]) if examples else 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.2 * max(rows, 1)))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, example in enumerate(examples):
        query_image, _ = raw_test[example["query_index"]]
        query_display = _to_display_image(query_image)
        axes[row, 0].imshow(query_display, cmap="gray" if query_display.ndim == 2 else None)
        axes[row, 0].set_title(f"Query: {example['query_label']}")
        axes[row, 0].axis("off")

        for col, support_index in enumerate(example["support_indices"], start=1):
            support_image, support_label = raw_test[support_index]
            distance = example["distances"][col - 1]
            support_display = _to_display_image(support_image)
            axes[row, col].imshow(support_display, cmap="gray" if support_display.ndim == 2 else None)
            axes[row, col].set_title(f"{support_label} | d={distance:.2f}")
            if support_label == example["prediction"]:
                color = "green" if example["correct"] else "red"
                for spine in axes[row, col].spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    _save_dual(output_path, report_path)
