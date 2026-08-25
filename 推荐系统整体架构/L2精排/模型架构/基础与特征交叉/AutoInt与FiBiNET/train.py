import torch
from torch.nn import functional as F

from model import AutoInt, FiBiNET


def build_dataset(size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    users = torch.randint(0, 32, (size,), generator=generator)
    items = torch.randint(0, 64, (size,), generator=generator)
    scenes = torch.randint(0, 4, (size,), generator=generator)
    labels = ((users % 8) == (items % 8)).float()
    labels = torch.maximum(labels, ((scenes == 3) & (users % 3 == 0)).float())
    return torch.stack([users, items, scenes], dim=1), labels


def train_model(model: torch.nn.Module, fields: torch.Tensor, labels: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(101):
        logits = model(fields)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"{model.__class__.__name__} epoch={epoch:03d} loss={loss.item():.4f}")
    with torch.no_grad():
        print(f"{model.__class__.__name__} probabilities={model(fields[:5]).sigmoid().tolist()}")


def main() -> None:
    fields, labels = build_dataset()
    field_sizes = [32, 64, 4]
    train_model(AutoInt(field_sizes), fields, labels)
    train_model(FiBiNET(field_sizes), fields, labels)


if __name__ == "__main__":
    main()