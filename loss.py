import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding_left: torch.Tensor,
        embedding_right: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        distances = F.pairwise_distance(embedding_left, embedding_right)
        positive_term = (1 - label) * distances.pow(2)
        negative_term = label * torch.clamp(self.margin - distances, min=0.0).pow(2)
        return torch.mean(positive_term + negative_term)
