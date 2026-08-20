import torch

from model import MatrixFactorization


def sample_triples(
    num_users: int, num_items: int, samples_per_user: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users, positives, negatives = [], [], []
    for user_id in range(num_users):
        positive_set = {(user_id * 5 + offset) % num_items for offset in range(3)}
        for sample_id in range(samples_per_user):
            positive = sorted(positive_set)[sample_id % len(positive_set)]
            negative = (positive + 7 + sample_id) % num_items
            while negative in positive_set:
                negative = (negative + 1) % num_items
            users.append(user_id)
            positives.append(positive)
            negatives.append(negative)
    return torch.tensor(users), torch.tensor(positives), torch.tensor(negatives)


def main() -> None:
    torch.manual_seed(7)
    num_users, num_items = 24, 96
    users, positives, negatives = sample_triples(num_users, num_items, 6)
    model = MatrixFactorization(num_users, num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    for epoch in range(101):
        loss = model.bpr_loss(users, positives, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    with torch.no_grad():
        user_ids = torch.zeros(num_items, dtype=torch.long)
        item_ids = torch.arange(num_items)
        top_items = model.score(user_ids, item_ids).topk(5).indices.tolist()
    print(f"user=0 top_items={top_items}")


if __name__ == "__main__":
    main()