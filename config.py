from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


SUPPORTED_DATASETS = (
    "mnist",
    "fashion_mnist",
    "fashion_products_small",
    "amharic_handwritten",
    "ethiopian_textile_local",
)

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "fashion_products_small": "Fashion Product Images Small",
    "amharic_handwritten": "Amharic Handwritten Characters",
    "ethiopian_textile_local": "Ethiopian Cultural Clothing and Textile",
}


@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    report_assets_dir: Path = field(init=False)

    seed: int = 42
    dataset_name: str = "mnist"
    image_size: Tuple[int, int] = (28, 28)
    input_channels: int = 1
    batch_size: int = 64
    num_workers: int = 0
    train_pairs_per_epoch: int = 12000
    val_pairs: int = 2000
    embedding_dim: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    margin: float = 1.0
    early_stopping_patience: int = 3
    one_shot_trials: int = 200
    one_shot_way: int = 5
    tsne_samples: int = 1000
    log_interval: int = 50
    model_name: str = field(init=False)
    history_name: str = field(init=False)
    metrics_name: str = field(init=False)
    results_table_name: str = field(init=False)
    sample_pairs_name: str = field(init=False)
    loss_curve_name: str = field(init=False)
    tsne_name: str = field(init=False)
    one_shot_examples_name: str = field(init=False)
    failures_name: str = field(init=False)
    report_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_name = self.dataset_name.lower().replace("-", "_")
        if self.dataset_name not in SUPPORTED_DATASETS:
            supported = ", ".join(SUPPORTED_DATASETS)
            raise ValueError(f"Unsupported dataset '{self.dataset_name}'. Choose from: {supported}.")
        self.data_dir = self.project_root / "data"
        self.outputs_dir = self.project_root / "outputs"
        self.figures_dir = self.outputs_dir / "figures"
        self.models_dir = self.outputs_dir / "models"
        self.results_dir = self.outputs_dir / "results"
        self.report_assets_dir = self.project_root / "report_assets"
        if self.dataset_name == "fashion_mnist":
            self.image_size = (28, 28)
            self.input_channels = 1
        elif self.dataset_name == "amharic_handwritten":
            self.image_size = (32, 32)
            self.input_channels = 1
        elif self.dataset_name == "fashion_products_small":
            self.image_size = (96, 96)
            self.input_channels = 3
        elif self.dataset_name == "ethiopian_textile_local":
            self.image_size = (128, 128)
            self.input_channels = 3
        suffix = self.dataset_name
        self.model_name = f"siamese_{suffix}.pt"
        self.history_name = f"training_history_{suffix}.json"
        self.metrics_name = f"evaluation_metrics_{suffix}.json"
        self.results_table_name = f"results_table_{suffix}.csv"
        self.sample_pairs_name = f"sample_pairs_{suffix}.png"
        self.loss_curve_name = f"loss_curve_{suffix}.png"
        self.tsne_name = f"tsne_plot_{suffix}.png"
        self.one_shot_examples_name = f"one_shot_examples_{suffix}.png"
        self.failures_name = f"one_shot_failure_analysis_{suffix}.csv"
        self.report_name = f"report_{suffix}.md"

    @property
    def dataset_title(self) -> str:
        return DATASET_LABELS[self.dataset_name]

    @property
    def class_names(self) -> List[str]:
        if self.dataset_name == "fashion_mnist":
            return [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        if self.dataset_name == "amharic_handwritten":
            return [str(index) for index in range(287)]
        if self.dataset_name == "fashion_products_small":
            return []
        if self.dataset_name == "ethiopian_textile_local":
            return []
        return [str(index) for index in range(10)]

    @property
    def normalization_stats(self) -> Tuple[Tuple[float], Tuple[float]]:
        if self.dataset_name == "fashion_mnist":
            return (0.2860,), (0.3530,)
        if self.dataset_name == "amharic_handwritten":
            return (0.5,), (0.5,)
        if self.dataset_name == "fashion_products_small":
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if self.dataset_name == "ethiopian_textile_local":
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return (0.1307,), (0.3081,)
