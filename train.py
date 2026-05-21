from typing import Dict, List, Tuple

import torch

from utils import save_json


def run_epoch(model, loader, criterion, optimizer, device, log_interval: int, training: bool) -> Tuple[float, float]:
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (left, right, pair_label, _, _) in enumerate(loader, start=1):
            left = left.to(device)
            right = right.to(device)
            pair_label = pair_label.to(device)

            if training:
                optimizer.zero_grad()

            left_embedding, right_embedding = model(left, right)
            loss = criterion(left_embedding, right_embedding, pair_label)

            if training:
                loss.backward()
                optimizer.step()

            distances = torch.nn.functional.pairwise_distance(left_embedding, right_embedding)
            predictions = (distances >= 0.5).float()
            correct += (predictions == pair_label).sum().item()
            total += pair_label.size(0)
            running_loss += loss.item() * pair_label.size(0)

            if training and step % log_interval == 0:
                print(f"Step {step:03d}/{len(loader):03d} - loss: {loss.item():.4f}")

    mean_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return mean_loss, accuracy


def train_model(config, model, train_loader, val_loader, criterion, optimizer, device) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_pair_accuracy": [],
        "val_pair_accuracy": [],
    }
    best_metrics = {
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_interval=config.log_interval,
            training=True,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_interval=config.log_interval,
            training=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pair_accuracy"].append(train_acc)
        history["val_pair_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch}/{config.epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- train_acc: {train_acc:.4f} "
            f"- val_acc: {val_acc:.4f}"
        )

        if val_loss < best_metrics["best_val_loss"]:
            best_metrics["best_val_loss"] = val_loss
            best_metrics["best_epoch"] = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.models_dir / config.model_name)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            print("Early stopping triggered.")
            break

    save_json(history, config.results_dir / config.history_name)
    return history, best_metrics
