import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 32, input_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.projection(self.features(x))
        return F.normalize(embedding, p=2, dim=1)


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 32, input_channels: int = 1) -> None:
        super().__init__()
        self.encoder = EmbeddingNetwork(embedding_dim=embedding_dim, input_channels=input_channels)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        left_embedding = self.forward_once(left)
        right_embedding = self.forward_once(right)
        return left_embedding, right_embedding
