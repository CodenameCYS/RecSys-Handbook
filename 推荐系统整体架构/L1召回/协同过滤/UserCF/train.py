from model import build_user_neighbors, recommend


def main() -> None:
    interactions = {
        1: {1: 1.0, 2: 1.0, 3: 1.0},
        2: {1: 1.0, 2: 1.0, 4: 2.0},
        3: {2: 1.0, 3: 1.0, 5: 1.5},
        4: {1: 1.0, 4: 1.0, 6: 1.0},
    }

    neighbors = build_user_neighbors(interactions, top_k=3, shrinkage=1.0)
    print("user=1 neighbors:", neighbors[1])
    print("user=1 recommendations:", recommend(interactions, neighbors, 1, top_k=5))


if __name__ == "__main__":
    main()