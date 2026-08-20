from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from annoy import AnnoyIndex


if __name__ == "__main__":
    generator = np.random.default_rng(7)
    dimension = 32
    item_ids = np.arange(50_000, 52_000, dtype=np.int64)
    item_vectors = generator.normal(size=(len(item_ids), dimension)).astype(np.float32)
    query_vector = item_vectors[789] + 0.01 * generator.normal(size=dimension)

    index = AnnoyIndex(dimension, metric="angular")
    for internal_id, vector in enumerate(item_vectors):
        index.add_item(internal_id, vector.tolist())
    index.build(n_trees=20, n_jobs=-1)

    with TemporaryDirectory() as directory:
        index_path = Path(directory) / "items.ann"
        index.save(str(index_path))

        restored = AnnoyIndex(dimension, metric="angular")
        restored.load(str(index_path))
        internal_ids, distances = restored.get_nns_by_vector(
            query_vector.tolist(),
            n=5,
            search_k=2_000,
            include_distances=True,
        )
        results = [
            (int(item_ids[internal_id]), float(distance))
            for internal_id, distance in zip(internal_ids, distances)
        ]
        print(results)
        restored.unload()