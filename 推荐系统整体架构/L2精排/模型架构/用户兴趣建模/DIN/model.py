import torch
from torch import nn


class ActivationUnit(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 4, 32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self, history: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        expanded_target = target.unsqueeze(1).expand_as(history)
        features = torch.cat(
            [history, expanded_target, history - expanded_target, history * expanded_target],
            dim=-1,
        )
        scores = self.network(features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=1)


class DIN(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.activation_unit = ActivationUnit(embedding_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 4, 64),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, history_item_ids: torch.Tensor, target_item_ids: torch.Tensor
    ) -> torch.Tensor:
        mask = history_item_ids.ne(0)
        history = self.item_embedding(history_item_ids)
        target = self.item_embedding(target_item_ids)
        attention = self.activation_unit(history, target, mask)
        interest = (attention.unsqueeze(-1) * history).sum(dim=1)
        features = torch.cat(
            [interest, target, interest - target, interest * target], dim=-1
        )
        return self.predictor(features).squeeze(-1)