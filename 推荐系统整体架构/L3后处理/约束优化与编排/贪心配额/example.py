from greedy_quota import Candidate, Quota, greedy_quota_select


candidates = [
    Candidate("tech_a", "tech", 0.98, 0),
    Candidate("tech_b", "tech", 0.94, 1),
    Candidate("life_a", "life", 0.73, 2),
    Candidate("life_b", "life", 0.70, 3),
]
quotas = [
    Quota("tech", maximum=2),
    Quota("life", minimum=1, maximum=2),
]

selected = greedy_quota_select(candidates, quotas, top_k=3)
print([(candidate.candidate_id, candidate.group) for candidate in selected])
