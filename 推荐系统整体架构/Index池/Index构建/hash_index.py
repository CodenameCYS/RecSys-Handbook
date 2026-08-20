from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class Candidate:
    item_id: int
    score: float
    reason: str


@dataclass(frozen=True)
class IndexEntry:
    candidates: tuple[Candidate, ...]
    source_version: str
    expires_at: float


class HashCandidateIndex:
    """A small in-memory ID-to-Item example with TTL and version metadata."""

    def __init__(self) -> None:
        self._entries: dict[str, IndexEntry] = {}

    def upsert(
        self,
        key: str,
        candidates: list[Candidate],
        source_version: str,
        ttl_seconds: float,
    ) -> None:
        deduplicated: dict[int, Candidate] = {}
        for candidate in candidates:
            current = deduplicated.get(candidate.item_id)
            if current is None or candidate.score > current.score:
                deduplicated[candidate.item_id] = candidate

        ranked = tuple(
            sorted(
                deduplicated.values(),
                key=lambda candidate: candidate.score,
                reverse=True,
            )
        )
        self._entries[key] = IndexEntry(
            candidates=ranked,
            source_version=source_version,
            expires_at=time() + ttl_seconds,
        )

    def get(self, key: str, top_k: int = 10) -> list[Candidate]:
        entry = self._entries.get(key)
        if entry is None:
            return []
        if entry.expires_at <= time():
            del self._entries[key]
            return []
        return list(entry.candidates[:top_k])

    def delete(self, key: str) -> None:
        self._entries.pop(key, None)


if __name__ == "__main__":
    index = HashCandidateIndex()
    index.upsert(
        key="item:100",
        candidates=[
            Candidate(201, 0.91, "co-occurrence"),
            Candidate(202, 0.87, "co-occurrence"),
            Candidate(201, 0.89, "duplicate-lower-score"),
        ],
        source_version="i2i_20260820",
        ttl_seconds=3600,
    )
    print(index.get("item:100", top_k=2))