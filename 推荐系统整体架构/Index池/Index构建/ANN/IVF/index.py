from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


class IVFIndex:
    def __init__(self, num_clusters: int = 32, seed: int = 7) -> None:
        self._num_clusters = num_clusters
        self._generator = np.random.default_rng(seed)
        self._centroids = np.empty((0, 0), dtype=np.float32)
        self._inverted_lists: dict[int, list[int]] = defaultdict(list)
        self._item_ids = np.empty(0, dtype=np.int64)
        self._vectors = np.empty((0, 0), dtype=np.float32)

    def train(self, vectors: NDArray[np.float32], iterations: int = 20) -> None:
        if len(vectors) < self._num_clusters:
            raise ValueError("training vectors must outnumber clusters")

        initial_positions = self._generator.choice(
            len(vectors), self._num_clusters, replace=False
        )
        centroids = vectors[initial_positions].astype(np.float32, copy=True)

        for _ in range(iterations):
            distances = np.sum(
                (vectors[:, None, :] - centroids[None, :, :]) ** 2,
                axis=2,
            )
            assignments = np.argmin(distances, axis=1)
            updated = centroids.copy()
            for cluster_id in range(self._num_clusters):
                members = vectors[assignments == cluster_id]
                if len(members):
                    updated[cluster_id] = members.mean(axis=0)
            if np.allclose(updated, centroids, atol=1e-4):
                break
            centroids = updated

        self._centroids = centroids

    def add(
        self,
        item_ids: NDArray[np.int64],
        vectors: NDArray[np.float32],
    ) -> None:
        if not len(self._centroids):
            raise RuntimeError("train the index before adding vectors")
        if len(item_ids) != len(vectors):
            raise ValueError("item_ids and vectors must have the same length")

        self._item_ids = item_ids.copy()
        self._vectors = vectors.astype(np.float32, copy=True)
        distances = np.sum(
            (self._vectors[:, None, :] - self._centroids[None, :, :]) ** 2,
            axis=2,
        )
        for position, cluster_id in enumerate(np.argmin(distances, axis=1)):
            self._inverted_lists[int(cluster_id)].append(position)

    def search(
        self,
        query: NDArray[np.float32],
        top_k: int,
        nprobe: int = 4,
    ) -> list[tuple[int, float]]:
        nprobe = min(max(nprobe, 1), self._num_clusters)
        centroid_distances = np.sum((self._centroids - query) ** 2, axis=1)
        selected_clusters = np.argpartition(centroid_distances, nprobe - 1)[:nprobe]
        positions = np.array(
            [
                position
                for cluster_id in selected_clusters
                for position in self._inverted_lists[int(cluster_id)]
            ],
            dtype=np.int64,
        )
        if not len(positions):
            return []

        distances = np.sum((self._vectors[positions] - query) ** 2, axis=1)
        result_count = min(top_k, len(positions))
        selected = np.argpartition(distances, result_count - 1)[:result_count]
        selected = selected[np.argsort(distances[selected])]
        return [
            (int(self._item_ids[positions[index]]), float(distances[index]))
            for index in selected
        ]


if __name__ == "__main__":
    generator = np.random.default_rng(7)
    item_ids = np.arange(30_000, 33_000, dtype=np.int64)
    item_vectors = generator.normal(size=(len(item_ids), 24)).astype(np.float32)
    query_vector = item_vectors[456] + 0.01 * generator.normal(size=24)

    index = IVFIndex(num_clusters=32)
    index.train(item_vectors)
    index.add(item_ids, item_vectors)
    print(index.search(query_vector.astype(np.float32), top_k=5, nprobe=4))