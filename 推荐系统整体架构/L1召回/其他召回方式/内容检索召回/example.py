from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt


@dataclass(frozen=True)
class ItemDocument:
    item_id: int
    category: str
    tokens: tuple[str, ...]


class TfidfRetriever:
    def __init__(self, documents: list[ItemDocument]) -> None:
        self.documents = documents
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document.tokens))
        count = len(documents)
        self.idf = {
            token: log((count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self.vectors = {document.item_id: self._vector(document.tokens) for document in documents}

    def _vector(self, tokens: tuple[str, ...]) -> dict[str, float]:
        frequencies = Counter(tokens)
        return {token: frequency * self.idf.get(token, 0.0) for token, frequency in frequencies.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def recall(
        self, query_tokens: tuple[str, ...], top_k: int, category: str | None = None
    ) -> list[tuple[int, float]]:
        query = self._vector(query_tokens)
        candidates = [
            (document.item_id, self._cosine(query, self.vectors[document.item_id]))
            for document in self.documents
            if category is None or document.category == category
        ]
        return sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:top_k]


if __name__ == "__main__":
    corpus = [
        ItemDocument(201, "shoe", ("running", "lightweight", "road")),
        ItemDocument(202, "shoe", ("running", "trail", "grip")),
        ItemDocument(203, "book", ("running", "training", "marathon")),
        ItemDocument(204, "shoe", ("casual", "leather", "city")),
    ]
    retriever = TfidfRetriever(corpus)
    print(retriever.recall(("running", "trail"), top_k=3, category="shoe"))