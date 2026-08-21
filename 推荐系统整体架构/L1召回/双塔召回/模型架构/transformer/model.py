import torch
from torch import nn
from torch.nn import functional as F


class TransformerTwoTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_categories: int,
        max_history_length: int = 8,
        embedding_dim: int = 32,
        output_dim: int = 24,
    ) -> None:
        super().__init__()
        self.max_history_length = max_history_length
        self.history_embeddings = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(max_history_length + 1, embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=embedding_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        self.user_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )
        self.user_projection = nn.Linear(embedding_dim, output_dim)

        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim)
        self.item_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim),
        )

    def encode_users(self, histories: torch.Tensor) -> torch.Tensor:
        if histories.size(1) > self.max_history_length:
            raise ValueError("history length exceeds max_history_length")
        batch_size, history_length = histories.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        history_vectors = self.history_embeddings(histories)
        tokens = torch.cat([cls_tokens, history_vectors], dim=1)
        positions = torch.arange(history_length + 1, device=histories.device)
        tokens = tokens + self.position_embeddings(positions).unsqueeze(0)

        history_padding = histories.eq(0)
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=histories.device)
        padding_mask = torch.cat([cls_padding, history_padding], dim=1)
        encoded = self.user_encoder(tokens, src_key_padding_mask=padding_mask)
        return F.normalize(self.user_projection(encoded[:, 0]), dim=-1)

    def encode_items(
        self, item_ids: torch.Tensor, category_ids: torch.Tensor
    ) -> torch.Tensor:
        item_features = torch.cat(
            [self.item_embeddings(item_ids), self.category_embeddings(category_ids)],
            dim=-1,
        )
        return F.normalize(self.item_projection(item_features), dim=-1)

    def in_batch_loss(
        self,
        histories: torch.Tensor,
        positive_item_ids: torch.Tensor,
        positive_category_ids: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        user_vectors = self.encode_users(histories)
        item_vectors = self.encode_items(positive_item_ids, positive_category_ids)
        logits = user_vectors @ item_vectors.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)