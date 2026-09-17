from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateState:
    in_stock: bool | None
    published: bool | None
    allowed_regions: frozenset[str] | None
    audience_allowed: bool | None
    rights_allowed: bool | None
    safety_approved: bool | None


@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    reason_codes: tuple[str, ...]


def evaluate_eligibility(
    state: CandidateState, request_region: str
) -> EligibilityDecision:
    reason_codes: list[str] = []
    checks = (
        (state.in_stock, "out_of_stock", "inventory_unknown"),
        (state.published, "not_published", "publish_state_unknown"),
        (
            None if state.allowed_regions is None else request_region in state.allowed_regions,
            "region_blocked",
            "region_policy_unknown",
        ),
        (state.audience_allowed, "audience_blocked", "audience_policy_unknown"),
        (state.rights_allowed, "rights_blocked", "rights_state_unknown"),
        (state.safety_approved, "safety_blocked", "safety_unknown"),
    )
    for passed, rejected_code, unknown_code in checks:
        if passed is False:
            reason_codes.append(rejected_code)
        elif passed is None:
            reason_codes.append(unknown_code)
    return EligibilityDecision(not reason_codes, tuple(reason_codes))