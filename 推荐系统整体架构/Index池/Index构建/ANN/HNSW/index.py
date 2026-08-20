from pathlib import Path
from tempfile import TemporaryDirectory

import hnswlib
import numpy as np


def build_index(vectors: np.ndarray, item_ids: np.ndarray) -> hnswlib.Index:
    dimension = vectors.shape[1]
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.init_index(
        max_elements=len(vectors),
        ef_construction=200,
        M=16,
        random_seed=7,
    )
    index.add_items(vectors, item_ids)
    index.set_ef(80)
    return index


if __name__ == "__main__":
    generator = np.random.default_rng(7)
    item_ids = np.arange(40_000, 42_000, dtype=np.int64)
    item_vectors = generator.normal(size=(len(item_ids), 32)).astype(np.float32)
    query_vector = item_vectors[678] + 0.01 * generator.normal(size=32)

    index = build_index(item_vectors, item_ids)
    labels, distances = index.knn_query(query_vector.astype(np.float32), k=5)
    print(list(zip(labels[0].tolist(), distances[0].tolist())))

    with TemporaryDirectory() as directory:
        index_path = Path(directory) / "items.hnsw"
        index.save_index(str(index_path))

        restored = hnswlib.Index(space="cosine", dim=item_vectors.shape[1])
        restored.load_index(str(index_path), max_elements=len(item_vectors))
        restored.set_ef(80)
        restored_labels, _ = restored.knn_query(query_vector.astype(np.float32), k=5)
        print("restored:", restored_labels[0].tolist())