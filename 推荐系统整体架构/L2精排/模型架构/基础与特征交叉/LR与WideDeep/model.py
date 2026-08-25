import torch
from torch import nn


class LogisticRegression(nn.Module):
    """Categorical-feature logistic regression with one embedding table per field."""

    def __init__(self, field_sizes: list[int]) -> None:
        super().__init__()
        self.linear_embeddings = nn.ModuleList(
            [nn.Embedding(field_size, 1) for field_size in field_sizes]
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        linear_terms = torch.stack(
            [
                embedding(fields[:, field_index])
                for field_index, embedding in enumerate(self.linear_embeddings)
            ],
            dim=1,
        )
        return (self.bias + linear_terms.sum(dim=1)).squeeze(-1)


class WideAndDeep(nn.Module):
    """Wide linear features plus a deep embedding MLP.

    Include manually engineered crosses as extra fields in ``fields``. The demo
    training script adds a user-group x item-category cross as the fourth field.
    """

    def __init__(self, field_sizes: list[int], embedding_dim: int = 8) -> None:
        super().__init__()
        self.wide_embeddings = nn.ModuleList(
            [nn.Embedding(field_size, 1) for field_size in field_sizes]
        )
        self.deep_embeddings = nn.ModuleList(
            [nn.Embedding(field_size, embedding_dim) for field_size in field_sizes]
        )
        input_dim = len(field_sizes) * embedding_dim
        self.deep_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        wide_terms = torch.stack(
            [
                embedding(fields[:, field_index])
                for field_index, embedding in enumerate(self.wide_embeddings)
            ],
            dim=1,
        ).sum(dim=1)
        deep_inputs = torch.stack(
            [
                embedding(fields[:, field_index])
                for field_index, embedding in enumerate(self.deep_embeddings)
            ],
            dim=1,
        ).flatten(start_dim=1)
        return (self.bias + wide_terms + self.deep_network(deep_inputs)).squeeze(-1)