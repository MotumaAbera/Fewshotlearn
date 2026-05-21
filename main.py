import argparse

from config import Config, SUPPORTED_DATASETS
from pipeline import prepare_environment, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot learning with Siamese Networks on MNIST.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override the learning rate.")
    parser.add_argument("--trials", type=int, default=None, help="Override the number of one-shot trials.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(SUPPORTED_DATASETS),
        help="Dataset to use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(dataset_name=args.dataset)

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.trials is not None:
        config.one_shot_trials = args.trials

    device = prepare_environment(config)
    print(f"Using device: {device}")
    results = run_experiment(config=config, device=device)
    metrics = results["metrics"]

    print("Training and evaluation complete.")
    print(f"One-shot accuracy: {metrics['one_shot_accuracy']:.4f}")


if __name__ == "__main__":
    main()
