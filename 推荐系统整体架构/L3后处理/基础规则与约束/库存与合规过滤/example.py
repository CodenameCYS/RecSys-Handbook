from eligibility_filter import CandidateState, evaluate_eligibility


def main() -> None:
    eligible_state = CandidateState(
        in_stock=True,
        published=True,
        allowed_regions=frozenset({"CN"}),
        audience_allowed=True,
        rights_allowed=True,
        safety_approved=True,
    )
    unknown_safety_state = CandidateState(
        in_stock=True,
        published=True,
        allowed_regions=frozenset({"CN"}),
        audience_allowed=True,
        rights_allowed=True,
        safety_approved=None,
    )
    print(evaluate_eligibility(eligible_state, request_region="CN"))
    print(evaluate_eligibility(unknown_safety_state, request_region="CN"))


if __name__ == "__main__":
    main()