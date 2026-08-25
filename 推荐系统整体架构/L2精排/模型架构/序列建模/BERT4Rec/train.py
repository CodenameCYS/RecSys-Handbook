import torch
from torch.nn import functional as F

from model import BERT4Rec


MASK_ITEM_ID = 101


def build_dataset(sample_count: int = 512, sequence_length: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    starts = torch.randint(1, 91, (sample_count,), generator=generator)
    offsets = torch.arange(sequence_length)
    sequences = ((starts.unsqueeze(1) + offsets - 1) % 100) + 1
    mask_positions = torch.randint(0, sequence_length, (sample_count,), generator=generator)
    masked_sequences = sequences.clone()
    masked_sequences[torch.arange(sample_count), mask_positions] = MASK_ITEM_ID
    labels = sequences[torch.arange(sample_count), mask_positions]
    return masked_sequences, mask_positions, labels


def main() -> None:
    torch.manual_seed(7)
    masked_sequences, mask_positions, labels = build_dataset()
    model = BERT4Rec(num_items=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(101):
        logits = model(masked_sequences)[torch.arange(labels.size(0)), mask_positions]
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")
    with torch.no_grad():
        predictions = model(masked_sequences[:5])[torch.arange(5), mask_positions[:5]].argmax(dim=-1).tolist()
    print(f"masked_item_predictions={predictions}")


if __name__ == "__main__":
    main()