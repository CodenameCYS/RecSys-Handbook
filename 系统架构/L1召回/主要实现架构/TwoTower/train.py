import torch

from model import TwoTowerModel


def build_training_pairs(num_users: int, num_items: int) -> tuple[torch.Tensor, torch.Tensor]:
    user_ids = torch.arange(num_users).repeat_interleave(2)
    offsets = torch.tensor([0, 1]).repeat(num_users)
    item_ids = (user_ids * 3 + offsets) % num_items
    return user_ids, item_ids


def main() -> None:
    torch.manual_seed(7)
    num_users, num_items = 32, 128
    users, positives = build_training_pairs(num_users, num_items)
    model = TwoTowerModel(num_users, num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(101):
        permutation = torch.randperm(users.size(0))
        total_loss = 0.0
        for start in range(0, users.size(0), 16):
            indices = permutation[start : start + 16]
            loss = model.in_batch_loss(users[indices], positives[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={total_loss:.4f}")

    with torch.no_grad():
        query = model.encode_users(torch.tensor([0]))
        item_vectors = model.encode_items(torch.arange(num_items))
        top_items = (query @ item_vectors.T).topk(5).indices.squeeze(0).tolist()
    print(f"user=0 top_items={top_items}")


if __name__ == "__main__":
    main()