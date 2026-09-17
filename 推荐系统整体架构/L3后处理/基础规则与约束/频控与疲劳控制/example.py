from frequency_cap import InMemoryExposureStore


def main() -> None:
    store = InMemoryExposureStore()
    for now in (100, 120, 140):
        allowed, reason_code = store.reserve(
            user_id="user_1",
            scope="creator:alice",
            now=now,
            window_seconds=300,
            max_exposures=2,
        )
        print(f"t={now}: allowed={allowed}, reason={reason_code}")


if __name__ == "__main__":
    main()