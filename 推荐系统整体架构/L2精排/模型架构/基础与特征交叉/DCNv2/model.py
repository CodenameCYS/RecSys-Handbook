import torch
from torch import nn


class CrossLayer(nn.Module):
    """A low-rank matrix cross layer from DCNv2."""

    def __init__(self, input_dim: int, low_rank: int) -> None:
        super().__init__()
        self.down_projection = nn.Linear(input_dim, low_rank, bias=False)
        self.up_projection = nn.Linear(low_rank, input_dim)

    def forward(self, base_inputs: torch.Tensor, cross_inputs: torch.Tensor) -> torch.Tensor:
        transformed = self.up_projection(self.down_projection(cross_inputs))
        return cross_inputs + base_inputs * transformed


class DCNv2(nn.Module):
    def __init__(
        self,
        field_sizes: list[int],
        embedding_dim: int = 8,
        cross_layers: int = 2,
        low_rank: int = 8,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(field_size, embedding_dim) for field_size in field_sizes]
        )
        input_dim = len(field_sizes) * embedding_dim
        self.cross_network = nn.ModuleList(
            [CrossLayer(input_dim, low_rank) for _ in range(cross_layers)]
        )
        self.deep_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim + 16, 1)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        base_inputs = torch.stack(
            [
                embedding(fields[:, field_index])
                for field_index, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        ).flatten(start_dim=1)
        cross_inputs = base_inputs
        for cross_layer in self.cross_network:
            cross_inputs = cross_layer(base_inputs, cross_inputs)
        deep_outputs = self.deep_network(base_inputs)
        return self.output(torch.cat([cross_inputs, deep_outputs], dim=1)).squeeze(-1)