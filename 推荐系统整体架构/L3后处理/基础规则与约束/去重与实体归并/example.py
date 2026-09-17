from deduplicate import Candidate, deduplicate_by_entity


def main() -> None:
    candidates = [
        Candidate("video_a", "article_101", 0.91, 0.80, 12, 0),
        Candidate("video_b", "article_101", 0.91, 0.85, 10, 1),
        Candidate("video_c", "article_202", 0.89, 0.75, 20, 2),
        Candidate("video_d", "article_303", 0.82, 0.90, 5, 3),
    ]
    kept, suppressed = deduplicate_by_entity(candidates)
    print("kept:", [candidate.item_id for candidate in kept])
    print("suppressed:", [(event.item_id, event.kept_item_id) for event in suppressed])


if __name__ == "__main__":
    main()