from random import Random

from epsilon_greedy import ActionEstimate, choose_action


def main() -> None:
    eligible_actions = [
        ActionEstimate(action_id="ranker_baseline", estimated_reward=0.42),
        ActionEstimate(action_id="ranker_diverse", estimated_reward=0.39),
        ActionEstimate(action_id="ranker_explore", estimated_reward=0.35),
    ]
    decision = choose_action(
        eligible_actions=eligible_actions,
        epsilon=0.15,
        random_source=Random(7),
    )
    print("decision:", decision)


if __name__ == "__main__":
    main()