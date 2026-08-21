import torch
from torch import nn
from torch.nn import functional as F


class DropoutItemEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 12) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(24, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(features), dim=-1)


def main() -> None:
    torch.manual_seed(7)
    num_items, num_categories = 24, 6
    categories = F.one_hot(torch.arange(num_items) // 4, num_categories).float()
    attributes = F.one_hot(torch.arange(num_items) % 4, 4).float()
    features = torch.cat([categories, attributes], dim=-1)
    model = DropoutItemEncoder(features.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(81):
        first_view = model(features)
        second_view = model(features)
        logits = first_view @ second_view.T / 0.1
        labels = torch.arange(num_items)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    vectors = model(features).detach()
    scores = vectors[0] @ vectors.T
    scores[0] = -1
    print("item=0 neighbors:", scores.topk(3).indices.tolist())


if __name__ == "__main__":
    main()