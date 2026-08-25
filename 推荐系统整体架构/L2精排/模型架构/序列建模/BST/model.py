import torch
from torch import nn


class BST(nn.Module):
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
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 4, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(
        self, history_item_ids: torch.Tensor, target_item_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size, history_length = history_item_ids.shape
        positions = torch.arange(history_length, device=history_item_ids.device)
        history = self.item_embedding(history_item_ids) + self.position_embedding(positions)
        padding_mask = history_item_ids.eq(0)
        encoded = self.encoder(history, src_key_padding_mask=padding_mask)
        valid_mask = (~padding_mask).unsqueeze(-1)
        sequence = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
        target = self.item_embedding(target_item_ids)
        features = torch.cat([sequence, target, sequence - target, sequence * target], dim=-1)
        return self.predictor(features).squeeze(-1)