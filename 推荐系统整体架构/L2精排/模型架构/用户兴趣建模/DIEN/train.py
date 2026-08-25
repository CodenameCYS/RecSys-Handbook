import torch
from torch.nn import functional as F

from model import DIEN


def build_dataset(sample_count: int = 1024, max_history_length: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    targets = torch.randint(1, 101, (sample_count,), generator=generator)
    lengths = torch.randint(2, max_history_length + 1, (sample_count,), generator=generator)
    histories = torch.zeros(sample_count, max_history_length, dtype=torch.long)
    negatives = torch.randint(1, 101, (sample_count, max_history_length), generator=generator)
    labels = torch.zeros(sample_count)
    for index, length in enumerate(lengths.tolist()):
        history = torch.randint(1, 101, (length,), generator=generator)
        if index % 2 == 0:
            history[-1] = ((targets[index] - 1) // 10) * 10 + 1
            labels[index] = 1.0
        histories[index, :length] = history
    return histories, targets, negatives, labels


def main() -> None:
    torch.manual_seed(7)
    histories, targets, negatives, labels = build_dataset()
    model = DIEN(num_items=101)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(101):
        logits = model(histories, targets)
        main_loss = F.binary_cross_entropy_with_logits(logits, labels)
        auxiliary_loss = model.auxiliary_loss(histories, negatives)
        loss = main_loss + 0.2 * auxiliary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f} main={main_loss.item():.4f} auxiliary={auxiliary_loss.item():.4f}")
    with torch.no_grad():
        print(f"probabilities={model(histories[:5], targets[:5]).sigmoid().tolist()}")


if __name__ == "__main__":
    main()