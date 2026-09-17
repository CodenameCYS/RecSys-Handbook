from dpp import Candidate, dpp_greedy_rerank


def main() -> None:
    candidates = [
        Candidate("tech_a", 0.95, [1.0, 0.0], 0),
        Candidate("tech_b", 0.93, [0.98, 0.02], 1),
        Candidate("life_a", 0.83, [0.0, 1.0], 2),
    ]
    result = dpp_greedy_rerank(candidates, top_k=2)
    print([candidate.item_id for candidate in result])


if __name__ == "__main__":
    main()