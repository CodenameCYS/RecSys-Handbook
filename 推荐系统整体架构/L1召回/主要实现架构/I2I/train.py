from model import ItemToItemCF


def main() -> None:
    sessions = [
        [1, 2, 3, 4],
        [1, 2, 5],
        [2, 3, 4],
        [6, 2, 3, 7],
        [1, 5, 8],
        [6, 7, 3],
    ]
    model = ItemToItemCF(window_size=2)
    model.fit(sessions)
    print(f"item=2 neighbors={model.neighbors[2][:5]}")
    print(f"seeds=[1, 2] recommendations={model.recommend([1, 2], top_k=5)}")


if __name__ == "__main__":
    main()