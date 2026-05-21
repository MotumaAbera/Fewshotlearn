from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch


def extract_embeddings(model, dataset, device, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings: List[np.ndarray] = []
    labels: List[int] = []

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    with torch.no_grad():
        for index in range(total):
            image, label = dataset[index]
            image = image.unsqueeze(0).to(device)
            embedding = model.forward_once(image).cpu().numpy()[0]
            embeddings.append(embedding)
            labels.append(int(label))

    return np.stack(embeddings), np.array(labels)


def run_one_shot_trials(model, dataset, device, trials: int, way: int, seed: int) -> Tuple[float, List[Dict[str, object]], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    effective_way = min(way, len(classes))
    support_lookup = {int(label): np.where(targets == label)[0] for label in classes}
    queryable_classes = [int(label) for label in classes if len(support_lookup[int(label)]) >= 2]
    if not queryable_classes:
        raise ValueError("One-shot evaluation requires at least one class with two or more examples.")
    examples: List[Dict[str, object]] = []
    records: List[Dict[str, object]] = []
    correct = 0

    model.eval()
    with torch.no_grad():
        for trial_index in range(trials):
            candidate_classes = rng.choice(classes, size=effective_way, replace=False)
            candidate_query_classes = [int(label) for label in candidate_classes if len(support_lookup[int(label)]) >= 2]
            if not candidate_query_classes:
                remaining = [label for label in queryable_classes if label not in candidate_classes]
                replacement = int(rng.choice(queryable_classes if not remaining else remaining))
                candidate_classes[0] = replacement
                candidate_query_classes = [replacement]
            query_class = int(rng.choice(candidate_query_classes))
            query_index = int(rng.choice(support_lookup[query_class]))
            support_indices = []

            for candidate_class in candidate_classes:
                support_index = query_index
                candidate_class = int(candidate_class)
                if candidate_class == query_class:
                    while support_index == query_index:
                        support_index = int(rng.choice(support_lookup[candidate_class]))
                else:
                    support_index = int(rng.choice(support_lookup[candidate_class]))
                support_indices.append(support_index)

            query_image, _ = dataset[query_index]
            query_embedding = model.forward_once(query_image.unsqueeze(0).to(device))

            distances = []
            for support_index in support_indices:
                support_image, support_label = dataset[support_index]
                support_embedding = model.forward_once(support_image.unsqueeze(0).to(device))
                distance = torch.nn.functional.pairwise_distance(query_embedding, support_embedding).item()
                distances.append((distance, int(support_label), support_index))

            predicted_label = min(distances, key=lambda item: item[0])[1]
            is_correct = predicted_label == query_class
            correct += int(is_correct)
            records.append(
                {
                    "trial": trial_index,
                    "query_index": query_index,
                    "query_label": query_class,
                    "predicted_label": predicted_label,
                    "correct": is_correct,
                    "closest_distance": min(distance for distance, _, _ in distances),
                    "support_labels": [label for _, label, _ in distances],
                    "support_indices": support_indices,
                }
            )

            if len(examples) < 6:
                examples.append(
                    {
                        "query_index": query_index,
                        "query_label": query_class,
                        "support_indices": support_indices,
                        "support_labels": [label for _, label, _ in distances],
                        "distances": [distance for distance, _, _ in distances],
                        "prediction": predicted_label,
                        "correct": is_correct,
                    }
                )

    return correct / max(trials, 1), examples, records


def analyze_one_shot_failures(records: List[Dict[str, object]], class_names: List[str]) -> Dict[str, object]:
    failures = [record for record in records if not record["correct"]]
    confusion = Counter((record["query_label"], record["predicted_label"]) for record in failures)
    confusion_rows = []
    for (query_label, predicted_label), count in confusion.most_common():
        confusion_rows.append(
            {
                "true_label": class_names[query_label],
                "predicted_label": class_names[predicted_label],
                "count": count,
            }
        )

    hardest_failures = sorted(failures, key=lambda item: item["closest_distance"])[:10]
    hardest_rows = []
    for failure in hardest_failures:
        hardest_rows.append(
            {
                "trial": failure["trial"],
                "query_index": failure["query_index"],
                "true_label": class_names[failure["query_label"]] if class_names else str(failure["query_label"]),
                "predicted_label": class_names[failure["predicted_label"]] if class_names else str(failure["predicted_label"]),
                "closest_distance": round(float(failure["closest_distance"]), 4),
            }
        )

    return {
        "failure_count": len(failures),
        "total_trials": len(records),
        "confusion_rows": confusion_rows,
        "hardest_rows": hardest_rows,
    }


def evaluate_model(config, model, raw_test, device, class_names: List[str]) -> Dict[str, object]:
    one_shot_accuracy, examples, records = run_one_shot_trials(
        model=model,
        dataset=raw_test,
        device=device,
        trials=config.one_shot_trials,
        way=config.one_shot_way,
        seed=config.seed + 20_000,
    )
    embeddings, labels = extract_embeddings(
        model=model,
        dataset=raw_test,
        device=device,
        max_samples=config.tsne_samples,
    )
    failure_analysis = analyze_one_shot_failures(records=records, class_names=class_names)

    return {
        "one_shot_accuracy": one_shot_accuracy,
        "effective_one_shot_way": min(config.one_shot_way, len(np.unique(np.array(raw_test.targets)))),
        "one_shot_examples": examples,
        "one_shot_records": records,
        "failure_analysis": failure_analysis,
        "embeddings": embeddings,
        "labels": labels,
    }
