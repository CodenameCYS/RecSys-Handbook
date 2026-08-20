import faiss
import numpy as np


if __name__ == "__main__":
    generator = np.random.default_rng(7)
    dimension = 32
    item_ids = np.arange(60_000, 65_000, dtype=np.int64)
    item_vectors = generator.normal(size=(len(item_ids), dimension)).astype(np.float32)
    faiss.normalize_L2(item_vectors)

    nlist = 32
    subquantizers = 8
    bits_per_code = 4
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        subquantizers,
        bits_per_code,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(item_vectors)
    index.add_with_ids(item_vectors, item_ids)
    index.nprobe = 8

    query_vector = item_vectors[890].reshape(1, -1).copy()
    scores, labels = index.search(query_vector, k=5)
    print(list(zip(labels[0].tolist(), scores[0].tolist())))