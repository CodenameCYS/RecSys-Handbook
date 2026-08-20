import torch

from model import ContentTwoTower


def main() -> None:
    torch.manual_seed(7)
    num_users = 24
    user_ids = torch.arange(num_users)
    positive_features = torch.stack([user_ids % 6, user_ids % 10], dim=1)
    model = ContentTwoTower(num_users, num_categories=6, num_creators=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for epoch in range(101):
        loss = model.contrastive_loss(user_ids, positive_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    new_item_features = torch.tensor([[0, 0], [0, 7], [4, 3], [5, 9]])
    with torch.no_grad():
        query = model.encode_users(torch.tensor([0]))
        scores = (query @ model.encode_items(new_item_features).T).squeeze(0)
    print(f"new_item_scores={scores.tolist()}")


if __name__ == "__main__":
    main()