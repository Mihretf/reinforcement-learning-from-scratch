import pickle
import numpy as np


RESULTS_PATH = "experiments/exploration_results.pkl"

WINDOW_SIZE = 100
SUCCESS_THRESHOLD = 0.90


with open(RESULTS_PATH, "rb") as file:
    results = pickle.load(file)


print("EXPLORATION CONVERGENCE ANALYSIS")
print("=" * 60)

print(
    f"Definition of convergence: "
    f"{SUCCESS_THRESHOLD:.0%} success over a "
    f"{WINDOW_SIZE}-episode rolling window."
)

print()


for strategy_name, data in results.items():

    successes = data["successes"]

    convergence_episode = None

    for i in range(
        WINDOW_SIZE,
        len(successes) + 1
    ):

        window = successes[
            i - WINDOW_SIZE:i
        ]

        success_rate = np.mean(window)

        if success_rate >= SUCCESS_THRESHOLD:

            convergence_episode = i
            break

    print(strategy_name)

    if convergence_episode is not None:

        print(
            f"Reached {SUCCESS_THRESHOLD:.0%} success at "
            f"episode {convergence_episode}"
        )

    else:

        print(
            f"Did not reach {SUCCESS_THRESHOLD:.0%} success."
        )

    print()


print("=" * 60)
print("Analysis complete.")