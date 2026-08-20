from rerank import mmr_rerank


def main() -> None:
    candidates = [
        {"item_id": 1, "score": 0.95, "vector": [1.0, 0.0]},
        {"item_id": 2, "score": 0.93, "vector": [0.98, 0.02]},
        {"item_id": 3, "score": 0.88, "vector": [0.0, 1.0]},
        {"item_id": 4, "score": 0.82, "vector": [0.3, 0.7]},
    ]
    result = mmr_rerank(candidates, top_k=3, relevance_weight=0.65)
    print([item["item_id"] for item in result])


if __name__ == "__main__":
    main()