import torch
from torch import nn
from torch.nn import functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        users = self.user_embeddings(user_ids)
        items = self.item_embeddings(item_ids)
        return (users * items).sum(dim=-1)

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
        regularization: float = 1e-4,
    ) -> torch.Tensor:
        positive_scores = self.score(user_ids, positive_item_ids)
        negative_scores = self.score(user_ids, negative_item_ids)
        ranking_loss = -F.logsigmoid(positive_scores - negative_scores).mean()
        regularizer = (
            self.user_embeddings(user_ids).square().mean()
            + self.item_embeddings(positive_item_ids).square().mean()
            + self.item_embeddings(negative_item_ids).square().mean()
        )
        return ranking_loss + regularization * regularizer