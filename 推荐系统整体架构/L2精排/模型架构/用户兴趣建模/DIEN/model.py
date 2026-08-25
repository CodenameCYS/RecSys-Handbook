import torch
from torch import nn
from torch.nn import functional as F


class AttentionScore(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(embedding_dim * 4, 32), nn.PReLU(), nn.Linear(32, 1))

    def forward(self, interests: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded_target = target.unsqueeze(1).expand_as(interests)
        features = torch.cat([interests, expanded_target, interests - expanded_target, interests * expanded_target], dim=-1)
        scores = self.network(features).squeeze(-1).masked_fill(~mask, torch.finfo(interests.dtype).min)
        return torch.softmax(scores, dim=1)


class AUGRUCell(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(embedding_dim, embedding_dim)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        proposal = self.cell(inputs, hidden)
        return hidden + attention.unsqueeze(-1) * (proposal - hidden)


class DIEN(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.interest_extractor = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.attention = AttentionScore(embedding_dim)
        self.evolution_cell = AUGRUCell(embedding_dim)
        self.auxiliary_network = nn.Sequential(nn.Linear(embedding_dim * 2, 32), nn.PReLU(), nn.Linear(32, 1))
        self.predictor = nn.Sequential(nn.Linear(embedding_dim * 4, 64), nn.PReLU(), nn.Linear(64, 1))

    def extract_interests(self, history_item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = history_item_ids.ne(0)
        interests, _ = self.interest_extractor(self.item_embedding(history_item_ids))
        return interests, mask

    def forward(self, history_item_ids: torch.Tensor, target_item_ids: torch.Tensor) -> torch.Tensor:
        interests, mask = self.extract_interests(history_item_ids)
        target = self.item_embedding(target_item_ids)
        attention = self.attention(interests, target, mask)
        evolved = torch.zeros_like(target)
        for step in range(history_item_ids.size(1)):
            updated = self.evolution_cell(interests[:, step], evolved, attention[:, step])
            evolved = torch.where(mask[:, step].unsqueeze(-1), updated, evolved)
        features = torch.cat([evolved, target, evolved - target, evolved * target], dim=-1)
        return self.predictor(features).squeeze(-1)

    def auxiliary_loss(self, history_item_ids: torch.Tensor, negative_item_ids: torch.Tensor) -> torch.Tensor:
        interests, mask = self.extract_interests(history_item_ids)
        valid_pairs = mask[:, :-1] & mask[:, 1:]
        states = interests[:, :-1]
        positive_items = self.item_embedding(history_item_ids[:, 1:])
        negative_items = self.item_embedding(negative_item_ids[:, 1:])
        positive_logits = self.auxiliary_network(torch.cat([states, positive_items], dim=-1)).squeeze(-1)
        negative_logits = self.auxiliary_network(torch.cat([states, negative_items], dim=-1)).squeeze(-1)
        positive_loss = F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits), reduction="none")
        negative_loss = F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits), reduction="none")
        return ((positive_loss + negative_loss) * valid_pairs).sum() / valid_pairs.sum().clamp_min(1)