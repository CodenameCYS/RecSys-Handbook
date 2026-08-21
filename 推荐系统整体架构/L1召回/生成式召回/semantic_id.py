from __future__ import annotations

from math import dist


class ResidualQuantizer:
    def __init__(self, codebooks: list[list[list[float]]]) -> None:
        self.codebooks = codebooks

    def encode(self, vector: list[float]) -> tuple[int, ...]:
        residual = vector[:]
        codes = []
        for codebook in self.codebooks:
            code = min(range(len(codebook)), key=lambda index: dist(residual, codebook[index]))
            codes.append(code)
            residual = [value - centroid for value, centroid in zip(residual, codebook[code], strict=True)]
        return tuple(codes)


class SemanticIdIndex:
    END = -1

    def __init__(self) -> None:
        self.root: dict[int, dict] = {}
        self.item_by_code: dict[tuple[int, ...], int] = {}

    def add(self, semantic_code: tuple[int, ...], item_id: int) -> tuple[int, ...]:
        collision_suffix = sum(code[: len(semantic_code)] == semantic_code for code in self.item_by_code)
        unique_code = (*semantic_code, collision_suffix, self.END)
        node = self.root
        for token in unique_code:
            node = node.setdefault(token, {})
        self.item_by_code[unique_code] = item_id
        return unique_code

    def allowed_tokens(self, prefix: tuple[int, ...]) -> list[int]:
        node = self.root
        for token in prefix:
            node = node.get(token, {})
        return sorted(node)

    def resolve(self, code: tuple[int, ...]) -> int | None:
        return self.item_by_code.get(code)