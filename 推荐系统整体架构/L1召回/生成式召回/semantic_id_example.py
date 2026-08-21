from semantic_id import ResidualQuantizer, SemanticIdIndex


def main() -> None:
    codebooks = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.2, 0.0], [0.0, 0.2]],
    ]
    item_vectors = {
        101: [1.1, 0.1],
        102: [1.0, 0.2],
        201: [0.1, 1.2],
    }
    quantizer = ResidualQuantizer(codebooks)
    index = SemanticIdIndex()
    codes = {
        item_id: index.add(quantizer.encode(vector), item_id)
        for item_id, vector in item_vectors.items()
    }
    print("semantic IDs:", codes)

    preferred_tokens = [0, 1, 0, SemanticIdIndex.END]
    prefix: tuple[int, ...] = ()
    while not prefix or prefix[-1] != SemanticIdIndex.END:
        allowed = index.allowed_tokens(prefix)
        token = next((candidate for candidate in preferred_tokens if candidate in allowed), allowed[0])
        prefix = (*prefix, token)
        print(f"prefix={prefix} allowed_before_step={allowed}")
    print("resolved item:", index.resolve(prefix))


if __name__ == "__main__":
    main()