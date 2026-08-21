import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveItemEncoder(nn.Module):
    def __init__(self, num_items: int, num_categories: int, dimension: int = 16) -> None:
        super().__init__()
        self.items = nn.Embedding(num_items, dimension)
        self.categories = nn.Embedding(num_categories, dimension)
        self.projection = nn.Linear(dimension * 2, dimension)

    def forward(self, item_ids: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.items(item_ids), self.categories(category_ids)], dim=-1)
        return F.normalize(self.projection(features), dim=-1)


def main() -> None:
    torch.manual_seed(7)
    num_items, num_categories = 24, 6
    anchors = torch.arange(0, num_items, 2)
    positives = anchors + 1
    categories = anchors // 4
    model = ContrastiveItemEncoder(num_items, num_categories)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(81):
        anchor_vectors = model(anchors, categories)
        positive_vectors = model(positives, categories)
        logits = anchor_vectors @ positive_vectors.T / 0.1
        labels = torch.arange(len(anchors))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    all_items = torch.arange(num_items)
    vectors = model(all_items, all_items // 4).detach()
    scores = vectors[0] @ vectors.T
    scores[0] = -1
    print("item=0 neighbors:", scores.topk(3).indices.tolist())


if __name__ == "__main__":
    main()