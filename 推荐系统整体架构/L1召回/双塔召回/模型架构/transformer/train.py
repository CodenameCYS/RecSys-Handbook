import torch

from model import TransformerTwoTower


def build_examples(
    num_examples: int, history_length: int, num_items: int, num_categories: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    histories = []
    targets = []
    for example_id in range(num_examples):
        category = example_id % num_categories
        sequence = [
            ((category + step * num_categories) % num_items) + 1
            for step in range(history_length + 1)
        ]
        histories.append(sequence[:-1])
        targets.append(sequence[-1])
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return (
        torch.tensor(histories, dtype=torch.long),
        target_tensor,
        (target_tensor - 1) % num_categories,
    )


def main() -> None:
    torch.manual_seed(7)
    num_items, num_categories, history_length = 80, 8, 5
    histories, positives, categories = build_examples(
        24, history_length, num_items, num_categories
    )
    model = TransformerTwoTower(
        num_items, num_categories, max_history_length=history_length
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(16):
        loss = model.in_batch_loss(histories, positives, categories)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        query = model.encode_users(histories[:1])
        item_ids = torch.arange(1, num_items + 1)
        item_categories = (item_ids - 1) % num_categories
        item_vectors = model.encode_items(item_ids, item_categories)
        top_items = item_ids[(query @ item_vectors.T).topk(5).indices[0]].tolist()
    print(f"history={histories[0].tolist()} top_items={top_items}")


if __name__ == "__main__":
    main()