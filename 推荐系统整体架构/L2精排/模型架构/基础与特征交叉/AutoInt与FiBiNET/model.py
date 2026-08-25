import torch
from torch import nn


class AutoInt(nn.Module):
    def __init__(
        self,
        field_sizes: list[int],
        embedding_dim: int = 8,
        attention_heads: int = 2,
        attention_layers: int = 2,
    ) -> None:
        super().__init__()
        if embedding_dim % attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(field_size, embedding_dim) for field_size in field_sizes]
        )
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embedding_dim, attention_heads, batch_first=True
                )
                for _ in range(attention_layers)
            ]
        )
        self.normalizations = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(attention_layers)]
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len(field_sizes) * embedding_dim, 1),
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        representations = torch.stack(
            [
                embedding(fields[:, field_index])
                for field_index, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )
        for attention_layer, normalization in zip(self.attention_layers, self.normalizations):
            attended, _ = attention_layer(representations, representations, representations)
            representations = normalization(representations + torch.relu(attended))
        return self.output(representations).squeeze(-1)


class FiBiNET(nn.Module):
    def __init__(self, field_sizes: list[int], embedding_dim: int = 8) -> None:
        super().__init__()
        self.field_count = len(field_sizes)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(field_size, embedding_dim) for field_size in field_sizes]
        )
        squeeze_dim = max(1, self.field_count // 2)
        self.squeeze_excitation = nn.Sequential(
            nn.Linear(self.field_count, squeeze_dim),
            nn.ReLU(),
            nn.Linear(squeeze_dim, self.field_count),
            nn.Sigmoid(),
        )
        self.bilinear_weight = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.bilinear_weight)
        interaction_count = self.field_count * (self.field_count - 1) // 2
        self.output = nn.Sequential(
            nn.Linear(interaction_count * embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _pairwise_interactions(self, embeddings: torch.Tensor) -> torch.Tensor:
        transformed = embeddings @ self.bilinear_weight
        interactions = []
        for left_index in range(self.field_count):
            for right_index in range(left_index + 1, self.field_count):
                interactions.append(
                    embeddings[:, left_index, :] * transformed[:, right_index, :]
                )
        return torch.cat(interactions, dim=1)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        embeddings = torch.stack(
            [
                embedding(fields[:, field_index])
                for field_index, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )
        squeeze_inputs = embeddings.mean(dim=2)
        field_weights = self.squeeze_excitation(squeeze_inputs).unsqueeze(-1)
        recalibrated_embeddings = embeddings * field_weights
        original_interactions = self._pairwise_interactions(embeddings)
        recalibrated_interactions = self._pairwise_interactions(recalibrated_embeddings)
        return self.output(
            torch.cat([original_interactions, recalibrated_interactions], dim=1)
        ).squeeze(-1)