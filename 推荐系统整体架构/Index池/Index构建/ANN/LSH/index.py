from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


def normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class RandomHyperplaneLSH:
    def __init__(
        self,
        dimension: int,
        num_tables: int = 6,
        num_bits: int = 12,
        seed: int = 7,
    ) -> None:
        generator = np.random.default_rng(seed)
        self._planes = generator.normal(
            size=(num_tables, num_bits, dimension)
        ).astype(np.float32)
        self._buckets: list[dict[int, list[int]]] = [
            defaultdict(list) for _ in range(num_tables)
        ]
        self._item_ids = np.empty(0, dtype=np.int64)
        self._vectors = np.empty((0, dimension), dtype=np.float32)

    @staticmethod
    def _signature(bits: NDArray[np.bool_]) -> int:
        signature = 0
        for bit_index, enabled in enumerate(bits):
            if enabled:
                signature |= 1 << bit_index
        return signature

    def add(
        self,
        item_ids: NDArray[np.int64],
        vectors: NDArray[np.float32],
    ) -> None:
        if len(item_ids) != len(vectors):
            raise ValueError("item_ids and vectors must have the same length")

        self._item_ids = item_ids.copy()
        self._vectors = normalize(vectors.astype(np.float32, copy=True))
        for table_index, planes in enumerate(self._planes):
            for position, vector in enumerate(self._vectors):
                signature = self._signature((planes @ vector) >= 0)
                self._buckets[table_index][signature].append(position)

    def search(
        self,
        query: NDArray[np.float32],
        top_k: int,
        candidate_target: int = 100,
    ) -> list[tuple[int, float]]:
        normalized_query = normalize(query.reshape(1, -1))[0]
        candidate_positions: set[int] = set()

        for table_index, planes in enumerate(self._planes):
            query_signature = self._signature((planes @ normalized_query) >= 0)
            table = self._buckets[table_index]
            candidate_positions.update(table.get(query_signature, []))

            if len(candidate_positions) < candidate_target:
                nearby_signatures = sorted(
                    table,
                    key=lambda signature: (signature ^ query_signature).bit_count(),
                )
                for signature in nearby_signatures:
                    candidate_positions.update(table[signature])
                    if len(candidate_positions) >= candidate_target:
                        break

        if not candidate_positions:
            return []

        positions = np.fromiter(candidate_positions, dtype=np.int64)
        scores = self._vectors[positions] @ normalized_query
        result_count = min(top_k, len(positions))
        selected = np.argpartition(scores, -result_count)[-result_count:]
        selected = selected[np.argsort(scores[selected])[::-1]]
        return [
            (int(self._item_ids[positions[index]]), float(scores[index]))
            for index in selected
        ]


if __name__ == "__main__":
    generator = np.random.default_rng(7)
    item_ids = np.arange(20_000, 22_000, dtype=np.int64)
    item_vectors = generator.normal(size=(len(item_ids), 32)).astype(np.float32)
    query_vector = item_vectors[321] + 0.02 * generator.normal(size=32)

    index = RandomHyperplaneLSH(dimension=32)
    index.add(item_ids, item_vectors)
    print(index.search(query_vector.astype(np.float32), top_k=5))