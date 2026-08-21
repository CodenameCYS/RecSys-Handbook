from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        interactions: list[tuple[int, int]],
        embedding_dim: int = 16,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)
        self.register_buffer(
            "normalized_adjacency",
            self._build_normalized_adjacency(interactions),
        )

    def _build_normalized_adjacency(
        self, interactions: list[tuple[int, int]]
    ) -> torch.Tensor:
        user_nodes = torch.tensor([user for user, _ in interactions])
        item_nodes = torch.tensor(
            [self.num_users + item for _, item in interactions]
        )
        sources = torch.cat((user_nodes, item_nodes))
        targets = torch.cat((item_nodes, user_nodes))
        num_nodes = self.num_users + self.num_items
        degrees = torch.bincount(sources, minlength=num_nodes).float()
        values = torch.rsqrt(degrees[sources] * degrees[targets])
        indices = torch.stack((sources, targets))
        return torch.sparse_coo_tensor(
            indices,
            values,
            (num_nodes, num_nodes),
            check_invariants=True,
        ).coalesce()

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat(
            (self.user_embeddings.weight, self.item_embeddings.weight), dim=0
        )
        layer_embeddings = [embeddings]
        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(self.normalized_adjacency, embeddings)
            layer_embeddings.append(embeddings)
        final_embeddings = torch.stack(layer_embeddings).mean(dim=0)
        return (
            final_embeddings[: self.num_users],
            final_embeddings[self.num_users :],
        )

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
        regularization: float = 1e-4,
    ) -> torch.Tensor:
        users, items = self.propagated_embeddings()
        positive_scores = (users[user_ids] * items[positive_item_ids]).sum(dim=-1)
        negative_scores = (users[user_ids] * items[negative_item_ids]).sum(dim=-1)
        ranking_loss = -F.logsigmoid(positive_scores - negative_scores).mean()
        regularizer = (
            self.user_embeddings(user_ids).square().mean()
            + self.item_embeddings(positive_item_ids).square().mean()
            + self.item_embeddings(negative_item_ids).square().mean()
        )
        return ranking_loss + regularization * regularizer

    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        excluded_items: set[int],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        users, items = self.propagated_embeddings()
        scores = users[user_id] @ items.T
        if excluded_items:
            excluded = torch.tensor(sorted(excluded_items), dtype=torch.long)
            scores[excluded] = -torch.inf
        values, indices = scores.topk(top_k)
        return list(zip(indices.tolist(), values.tolist(), strict=True))