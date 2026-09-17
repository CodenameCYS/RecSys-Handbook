from xquad import Candidate, xquad_rerank


def main() -> None:
    candidates = [
        Candidate("tech_a", 0.95, {"tech": 0.95, "life": 0.05}, 0),
        Candidate("tech_b", 0.92, {"tech": 0.90, "life": 0.10}, 1),
        Candidate("life_a", 0.82, {"tech": 0.05, "life": 0.95}, 2),
    ]
    result = xquad_rerank(
        candidates,
        intent_weights={"tech": 0.55, "life": 0.45},
        top_k=2,
        diversity_weight=0.6,
    )
    print([candidate.item_id for candidate in result])


if __name__ == "__main__":
    main()