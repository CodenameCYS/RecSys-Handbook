from sampling import (
    ExposureSampler,
    PopularitySampler,
    UniformSampler,
    corrected_logits,
    sample_histogram,
)


def main() -> None:
    item_ids = list(range(1, 9))
    frequencies = {item: item * item for item in item_ids}
    excluded = {1, 2}

    uniform = UniformSampler(item_ids)
    popularity = PopularitySampler(frequencies)
    exposure = ExposureSampler()

    uniform_samples = uniform.sample(positive_item=1, excluded_items=excluded, count=4)
    popularity_samples = popularity.sample(
        positive_item=1, excluded_items=excluded, count=200
    )
    exposure_samples = exposure.sample_from_exposures(
        exposed_items=[1, 3, 4, 5], positive_items={1}, count=2
    )

    print("uniform:", uniform_samples)
    print("popularity histogram:", sample_histogram(popularity_samples))
    print("exposure:", exposure_samples)

    raw_scores = {sample.item_id: 1.0 for sample in popularity_samples}
    adjusted = corrected_logits(raw_scores, popularity_samples)
    print("corrected logits:", sorted(adjusted.items()))


if __name__ == "__main__":
    main()