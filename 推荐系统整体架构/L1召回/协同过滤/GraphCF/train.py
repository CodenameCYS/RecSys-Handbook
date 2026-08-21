import torch

from model import LightGCN


def build_interactions(num_users: int, num_items: int) -> list[tuple[int, int]]:
    return [
        (user_id, (user_id * 5 + offset) % num_items)
        for user_id in range(num_users)
        for offset in range(3)
    ]


def sample_triples(
    interactions: list[tuple[int, int]], num_users: int, num_items: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_by_user = {user_id: set() for user_id in range(num_users)}
    for user_id, item_id in interactions:
        positive_by_user[user_id].add(item_id)

    users, positives, negatives = [], [], []
    for sample_id, (user_id, positive) in enumerate(interactions):
        negative = (positive + 7 + sample_id) % num_items
        while negative in positive_by_user[user_id]:
            negative = (negative + 1) % num_items
        users.append(user_id)
        positives.append(positive)
        negatives.append(negative)
    return torch.tensor(users), torch.tensor(positives), torch.tensor(negatives)


def main() -> None:
    torch.manual_seed(7)
    num_users, num_items = 24, 96
    interactions = build_interactions(num_users, num_items)
    users, positives, negatives = sample_triples(
        interactions, num_users, num_items
    )
    model = LightGCN(num_users, num_items, interactions)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    for epoch in range(101):
        loss = model.bpr_loss(users, positives, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    history = {item for user, item in interactions if user == 0}
    print("user=0 history:", sorted(history))
    print("user=0 new recommendations:", model.recommend(0, history, top_k=5))


if __name__ == "__main__":
    main()