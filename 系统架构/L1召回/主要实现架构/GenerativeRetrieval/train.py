import torch

from model import GenerativeRecallModel


def build_sequences(num_sequences: int, sequence_length: int, num_items: int) -> torch.Tensor:
    rows = []
    for start in range(1, num_sequences + 1):
        rows.append([((start + step * 3) % (num_items - 1)) + 1 for step in range(sequence_length)])
    return torch.tensor(rows, dtype=torch.long)


def main() -> None:
    torch.manual_seed(7)
    num_items = 64
    sequences = build_sequences(96, 6, num_items)
    model = GenerativeRecallModel(num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(101):
        loss = model.next_item_loss(sequences)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    history = sequences[0, :4].unsqueeze(0)
    print(f"history={history.squeeze(0).tolist()} next_items={model.next_items(history)}")


if __name__ == "__main__":
    main()