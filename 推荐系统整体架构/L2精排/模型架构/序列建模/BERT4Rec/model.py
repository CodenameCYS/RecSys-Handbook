import torch
from torch import nn


class BERT4Rec(nn.Module):
    def __init__(
        self, num_items: int, embedding_dim: int = 16, max_length: int = 8
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 2, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Linear(embedding_dim, num_items + 1)

    def forward(self, masked_item_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = masked_item_ids.shape
        positions = torch.arange(sequence_length, device=masked_item_ids.device)
        inputs = self.item_embedding(masked_item_ids) + self.position_embedding(positions)
        encoded = self.encoder(inputs, src_key_padding_mask=masked_item_ids.eq(0))
        return self.output(encoded)