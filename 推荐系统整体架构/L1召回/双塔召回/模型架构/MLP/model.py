import torch
from torch import nn
from torch.nn import functional as F


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        output_dim: int = 16,
    ) -> None:
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Embedding(num_users, embedding_dim),
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )
        self.item_tower = nn.Sequential(
            nn.Embedding(num_items, embedding_dim),
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(user_ids), dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_tower(item_ids), dim=-1)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_users(user_ids), self.encode_items(item_ids)

    def in_batch_loss(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        user_vectors, item_vectors = self(user_ids, positive_item_ids)
        logits = user_vectors @ item_vectors.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)