from integer_programming import Candidate as IpCandidate
from integer_programming import Quota, solve_assignment
from lagrangian import Candidate as DualCandidate
from lagrangian import optimize_soft_caps


raw_candidates = [
    ("news_a", "news", 0.96, 0),
    ("news_b", "news", 0.92, 1),
    ("sports_a", "sports", 0.80, 2),
    ("culture_a", "culture", 0.78, 3),
]

try:
    assignment = solve_assignment(
        candidates=[
            IpCandidate(candidate_id, group, utility, frozenset({0, 1, 2}), input_index)
            for candidate_id, group, utility, input_index in raw_candidates
        ],
        quotas=[
            Quota("news", maximum=2),
            Quota("sports", minimum=1, maximum=2),
            Quota("culture", maximum=2),
        ],
        position_weights=[100, 80, 60],
    )
except RuntimeError as error:
    print("cp_sat: unavailable", error)
else:
    if assignment is None:
        raise RuntimeError("the example assignment should be feasible")
    print("cp_sat:", [(item.position, item.candidate_id) for item in assignment.placements])
    print("cp_sat_status:", assignment.status)

dual_result = optimize_soft_caps(
    candidates=[
        DualCandidate(candidate_id, group, utility, input_index)
        for candidate_id, group, utility, input_index in raw_candidates
    ],
    target_counts={"news": 1},
    top_k=3,
)
print("lagrangian:", dual_result.selected_ids)
print("multipliers:", dual_result.multipliers)
