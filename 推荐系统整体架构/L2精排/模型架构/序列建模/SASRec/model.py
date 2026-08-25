import torch
from torch import nn


class SASRec(nn.Module):
    def __init__(
        self, num_items: int, embedding_dim: int = 16, max_length: int = 8
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Linear(embedding_dim, num_items)

    def forward(self, sequence_item_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = sequence_item_ids.shape
        positions = torch.arange(sequence_length, device=sequence_item_ids.device)
        inputs = self.item_embedding(sequence_item_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=sequence_item_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(
            inputs, mask=causal_mask, src_key_padding_mask=sequence_item_ids.eq(0)
        )
        return self.output(encoded)