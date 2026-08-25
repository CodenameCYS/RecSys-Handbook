import torch
from torch.nn import functional as F

from model import SASRec


def build_dataset(sample_count: int = 512, sequence_length: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    starts = torch.randint(1, 91, (sample_count,), generator=generator)
    offsets = torch.arange(sequence_length)
    return ((starts.unsqueeze(1) + offsets - 1) % 100) + 1


def main() -> None:
    torch.manual_seed(7)
    sequences = build_dataset()
    model = SASRec(num_items=101)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(101):
        logits = model(sequences[:, :-1])
        loss = F.cross_entropy(logits.flatten(end_dim=1), sequences[:, 1:].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")
    with torch.no_grad():
        predictions = model(sequences[:5, :-1])[:, -1].argmax(dim=-1).tolist()
    print(f"next_item_predictions={predictions}")


if __name__ == "__main__":
    main()