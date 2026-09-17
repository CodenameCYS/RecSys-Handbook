from rolling_exposure import Candidate, calculate_adjustments, rerank_with_exposure_adjustment


def main() -> None:
    weighted_exposure = {"head": 90.0, "tail": 10.0}
    target_share = {"head": 0.7, "tail": 0.3}
    adjustments = calculate_adjustments(
        weighted_exposure=weighted_exposure,
        target_share=target_share,
        learning_rate=0.5,
        max_adjustment=0.15,
    )
    candidates = [
        Candidate(item_id="head_a", group_id="head", utility=0.90, input_index=0),
        Candidate(item_id="tail_a", group_id="tail", utility=0.82, input_index=1),
        Candidate(item_id="head_b", group_id="head", utility=0.79, input_index=2),
    ]

    reranked = rerank_with_exposure_adjustment(candidates, adjustments)
    print("adjustments:", adjustments)
    print("reranked:", [(candidate.item_id, candidate.adjusted_score) for candidate in reranked])


if __name__ == "__main__":
    main()