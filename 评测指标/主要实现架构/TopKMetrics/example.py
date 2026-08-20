from metrics import macro_average


def main() -> None:
    recommendations = {1: [8, 3, 2, 7], 2: [4, 6, 9, 1]}
    ground_truth = {1: {2, 3}, 2: {1}}
    print(macro_average(recommendations, ground_truth, top_k=3))


if __name__ == "__main__":
    main()