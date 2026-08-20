import numpy as np
from numpy.typing import NDArray


def normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class FlatCosineIndex:
    def __init__(self) -> None:
        self._item_ids = np.empty(0, dtype=np.int64)
        self._vectors = np.empty((0, 0), dtype=np.float32)

    def add(
        self,
        item_ids: NDArray[np.int64],
        vectors: NDArray[np.float32],
    ) -> None:
        if len(item_ids) != len(vectors):
            raise ValueError("item_ids and vectors must have the same length")
        self._item_ids = item_ids.copy()
        self._vectors = normalize(vectors.astype(np.float32, copy=True))

    def search(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> list[tuple[int, float]]:
        if not 0 < top_k <= len(self._item_ids):
            raise ValueError("top_k must be between 1 and the index size")

        normalized_query = normalize(query.reshape(1, -1))[0]
        scores = self._vectors @ normalized_query
        top_positions = np.argpartition(scores, -top_k)[-top_k:]
        top_positions = top_positions[np.argsort(scores[top_positions])[::-1]]
        return [
            (int(self._item_ids[position]), float(scores[position]))
            for position in top_positions
        ]


if __name__ == "__main__":
    generator = np.random.default_rng(7)
    item_ids = np.arange(10_000, 11_000, dtype=np.int64)
    item_vectors = generator.normal(size=(len(item_ids), 32)).astype(np.float32)
    query_vector = item_vectors[123] + 0.01 * generator.normal(size=32)

    index = FlatCosineIndex()
    index.add(item_ids, item_vectors)
    print(index.search(query_vector.astype(np.float32), top_k=5))