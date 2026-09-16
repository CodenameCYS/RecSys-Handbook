import torch
from torch.nn import functional as F

from model import PLE


def build_dataset(size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(size, 8, generator=generator)
    click_labels = ((features[:, 0] + features[:, 1]) > 0).float()
    value_labels = ((features[:, 2] - features[:, 3] + 0.4 * features[:, 0]) > 0).float()
    return features, torch.stack([click_labels, value_labels], dim=1)


def main() -> None:
    torch.manual_seed(7)
    features, labels = build_dataset()
    model = PLE(input_dim=features.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(101):
        logits = model(features)
        task_losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=0)
        loss = task_losses.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f} click={task_losses[0].item():.4f} value={task_losses[1].item():.4f}")
    with torch.no_grad():
        print(f"task_probabilities={model(features[:5]).sigmoid().tolist()}")


if __name__ == "__main__":
    main()