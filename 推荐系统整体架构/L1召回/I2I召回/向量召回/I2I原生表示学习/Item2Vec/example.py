import random

import torch
from torch import nn
from torch.nn import functional as F


class Item2Vec(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 12) -> None:
        super().__init__()
        self.center = nn.Embedding(num_items, embedding_dim)
        self.context = nn.Embedding(num_items, embedding_dim)

    def loss(
        self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        center_vectors = self.center(centers)
        positive_logits = (center_vectors * self.context(contexts)).sum(dim=-1)
        negative_vectors = self.context(negatives)
        negative_logits = torch.einsum("bd,bkd->bk", center_vectors, negative_vectors)
        return -(
            F.logsigmoid(positive_logits)
            + F.logsigmoid(-negative_logits).sum(dim=-1)
        ).mean()


def build_pairs(sessions: list[list[int]], window: int = 2) -> list[tuple[int, int]]:
    pairs = []
    for session in sessions:
        for index, center in enumerate(session):
            left, right = max(0, index - window), min(len(session), index + window + 1)
            pairs.extend((center, context) for context in session[left:right] if context != center)
    return pairs


def main() -> None:
    torch.manual_seed(7)
    random.seed(7)
    sessions = [[1, 2, 3, 4], [1, 2, 3, 5], [6, 7, 8, 9], [6, 7, 8, 10]] * 8
    pairs = build_pairs(sessions)
    centers = torch.tensor([pair[0] for pair in pairs])
    contexts = torch.tensor([pair[1] for pair in pairs])
    negatives = torch.randint(1, 11, (len(pairs), 4))
    model = Item2Vec(num_items=11)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(61):
        loss = model.loss(centers, contexts, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    vectors = F.normalize(model.center.weight.detach(), dim=-1)
    similarities = vectors[1] @ vectors.T
    similarities[0:2] = -1
    print("item=1 neighbors:", similarities.topk(3).indices.tolist())


if __name__ == "__main__":
    main()