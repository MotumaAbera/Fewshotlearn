from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import zipfile


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_dataset: "SiamesePairDataset"
    val_dataset: "SiamesePairDataset"
    raw_train: Dataset
    raw_test: Dataset
    class_names: List[str]


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform, label_names: List[str]) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.label_names = label_names
        self.targets = np.array([int(item["label"]) for item in hf_dataset], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int):
        item = self.hf_dataset[index]
        image = item["image"].convert("RGB")
        label = int(item["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class SiamesePairDataset(Dataset):
    def __init__(
        self,
        image_dataset: Dataset,
        pairs_per_epoch: int,
        seed: int,
    ) -> None:
        self.image_dataset = image_dataset
        self.pairs_per_epoch = pairs_per_epoch
        self.seed = seed
        self.targets = np.array(image_dataset.targets)
        self.label_to_indices = self._build_index(self.targets)
        self.labels = sorted(self.label_to_indices.keys())
        self.labels_with_multiple_examples = [
            label for label, indices in self.label_to_indices.items() if len(indices) >= 2
        ]

    @staticmethod
    def _build_index(targets: Sequence[int]) -> Dict[int, np.ndarray]:
        lookup: Dict[int, List[int]] = defaultdict(list)
        for index, label in enumerate(targets):
            lookup[int(label)].append(index)
        return {label: np.array(indices, dtype=np.int64) for label, indices in lookup.items()}

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + index)
        anchor_index = int(rng.integers(0, len(self.image_dataset)))
        anchor_image, anchor_label = self.image_dataset[anchor_index]

        same_class = bool(rng.integers(0, 2))
        if same_class and len(self.label_to_indices[int(anchor_label)]) >= 2:
            partner_index = anchor_index
            while partner_index == anchor_index:
                partner_index = int(rng.choice(self.label_to_indices[int(anchor_label)]))
            label = 0.0
        else:
            negative_label = int(rng.choice([item for item in self.labels if item != int(anchor_label)]))
            partner_index = int(rng.choice(self.label_to_indices[negative_label]))
            label = 1.0

        partner_image, partner_label = self.image_dataset[partner_index]
        return (
            anchor_image,
            partner_image,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(anchor_label, dtype=torch.long),
            torch.tensor(partner_label, dtype=torch.long),
        )


def get_transforms(config):
    mean, std = config.normalization_stats
    transform_steps = [transforms.Resize(config.image_size)]
    if config.input_channels == 1:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(transform_steps)


def get_dataset_class(config):
    return datasets.FashionMNIST if config.dataset_name == "fashion_mnist" else datasets.MNIST


def download_and_extract_geez_dataset(config) -> Path:
    target_root = config.data_dir / "amharic_handwritten"
    extract_root = target_root / "OCR_dataset"
    if extract_root.exists():
        return extract_root
    target_root.mkdir(parents=True, exist_ok=True)
    archive_path = hf_hub_download(
        repo_id="Yaredoffice/geez-characters",
        filename="OCR_dataset.zip",
        repo_type="dataset",
        local_dir=target_root,
    )
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(target_root)
    return extract_root


def create_amharic_dataset_bundle(config) -> DatasetBundle:
    transform = get_transforms(config)
    dataset_root = download_and_extract_geez_dataset(config)
    raw_train = ImageFolder(root=dataset_root / "train", transform=transform)
    raw_test = ImageFolder(root=dataset_root / "test", transform=transform)
    class_names = list(raw_train.classes)

    train_dataset = SiamesePairDataset(
        image_dataset=raw_train,
        pairs_per_epoch=config.train_pairs_per_epoch,
        seed=config.seed,
    )
    val_dataset = SiamesePairDataset(
        image_dataset=raw_test,
        pairs_per_epoch=config.val_pairs,
        seed=config.seed + 10_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        raw_train=raw_train,
        raw_test=raw_test,
        class_names=class_names,
    )


def create_ethiopian_textile_bundle(config) -> DatasetBundle:
    transform = get_transforms(config)
    dataset_root = config.data_dir / "ethiopian_textile_local"
    train_root = dataset_root / "train"
    test_root = dataset_root / "test"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            "Expected a local textile dataset at data/ethiopian_textile_local/train and "
            "data/ethiopian_textile_local/test. See the README for the required folder structure."
        )
    raw_train = ImageFolder(root=train_root, transform=transform)
    raw_test = ImageFolder(root=test_root, transform=transform)
    class_names = list(raw_train.classes)

    train_dataset = SiamesePairDataset(
        image_dataset=raw_train,
        pairs_per_epoch=config.train_pairs_per_epoch,
        seed=config.seed,
    )
    val_dataset = SiamesePairDataset(
        image_dataset=raw_test,
        pairs_per_epoch=config.val_pairs,
        seed=config.seed + 10_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        raw_train=raw_train,
        raw_test=raw_test,
        class_names=class_names,
    )


def create_fashion_products_bundle(config) -> DatasetBundle:
    transform = get_transforms(config)
    dataset = hf_load_dataset("ashraq/fashion-product-images-small", split="train")
    dataset = dataset.filter(lambda item: item["masterCategory"] == "Apparel")

    article_counts: Dict[str, int] = defaultdict(int)
    for item in dataset:
        article_counts[item["articleType"]] += 1

    top_classes = [name for name, _ in sorted(article_counts.items(), key=lambda pair: pair[1], reverse=True)[:8]]
    class_to_idx = {name: idx for idx, name in enumerate(top_classes)}
    dataset = dataset.filter(lambda item: item["articleType"] in class_to_idx)

    def relabel(item):
        item["label"] = class_to_idx[item["articleType"]]
        return item

    dataset = dataset.map(relabel)
    split = dataset.train_test_split(test_size=0.2, seed=config.seed)
    raw_train = HFDatasetWrapper(split["train"], transform=transform, label_names=top_classes)
    raw_test = HFDatasetWrapper(split["test"], transform=transform, label_names=top_classes)

    train_dataset = SiamesePairDataset(
        image_dataset=raw_train,
        pairs_per_epoch=config.train_pairs_per_epoch,
        seed=config.seed,
    )
    val_dataset = SiamesePairDataset(
        image_dataset=raw_test,
        pairs_per_epoch=config.val_pairs,
        seed=config.seed + 10_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        raw_train=raw_train,
        raw_test=raw_test,
        class_names=top_classes,
    )


def create_datasets(config) -> DatasetBundle:
    if config.dataset_name == "amharic_handwritten":
        return create_amharic_dataset_bundle(config)
    if config.dataset_name == "fashion_products_small":
        return create_fashion_products_bundle(config)
    if config.dataset_name == "ethiopian_textile_local":
        return create_ethiopian_textile_bundle(config)

    transform = get_transforms(config)
    dataset_class = get_dataset_class(config)
    raw_train = dataset_class(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    raw_test = dataset_class(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = SiamesePairDataset(
        image_dataset=raw_train,
        pairs_per_epoch=config.train_pairs_per_epoch,
        seed=config.seed,
    )
    val_dataset = SiamesePairDataset(
        image_dataset=raw_test,
        pairs_per_epoch=config.val_pairs,
        seed=config.seed + 10_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        raw_train=raw_train,
        raw_test=raw_test,
        class_names=config.class_names,
    )
