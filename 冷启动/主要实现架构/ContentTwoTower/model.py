import torch
from torch import nn
from torch.nn import functional as F


class ContentTwoTower(nn.Module):
    def __init__(
        self, num_users: int, num_categories: int, num_creators: int, embedding_dim: int = 16
    ) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim)
        self.creator_embeddings = nn.Embedding(num_creators, embedding_dim)
        self.item_projection = nn.Linear(embedding_dim * 2, embedding_dim)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_embeddings(user_ids), dim=-1)

    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        categories = self.category_embeddings(item_features[:, 0])
        creators = self.creator_embeddings(item_features[:, 1])
        return F.normalize(self.item_projection(torch.cat([categories, creators], dim=-1)), dim=-1)

    def contrastive_loss(
        self, user_ids: torch.Tensor, positive_item_features: torch.Tensor, temperature: float = 0.1
    ) -> torch.Tensor:
        users = self.encode_users(user_ids)
        items = self.encode_items(positive_item_features)
        logits = users @ items.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)