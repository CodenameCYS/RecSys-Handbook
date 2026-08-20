import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(self, field_sizes: list[int], embedding_dim: int = 8) -> None:
        super().__init__()
        self.linear_embeddings = nn.ModuleList([nn.Embedding(size, 1) for size in field_sizes])
        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in field_sizes]
        )
        input_dim = len(field_sizes) * embedding_dim
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        linear = torch.stack(
            [embedding(fields[:, index]) for index, embedding in enumerate(self.linear_embeddings)],
            dim=1,
        ).sum(dim=1)
        embeddings = torch.stack(
            [embedding(fields[:, index]) for index, embedding in enumerate(self.feature_embeddings)],
            dim=1,
        )
        summed = embeddings.sum(dim=1)
        fm = 0.5 * (summed.square() - embeddings.square().sum(dim=1)).sum(dim=1, keepdim=True)
        deep = self.deep(embeddings.flatten(start_dim=1))
        return (self.bias + linear + fm + deep).squeeze(-1)