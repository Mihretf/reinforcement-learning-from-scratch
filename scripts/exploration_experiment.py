
import pickle

import numpy as np
import matplotlib.pyplot as plt

from environment.grid_world import GridWorld
from scripts.q_learning import QLearningAgent


NUM_EPISODES = 1000
MAX_STEPS = 100

WINDOW_SIZE = 100

EXPLORATION_STRATEGIES = {
    "High Exploration": 1.0,
    "Moderate Exploration": 0.5,
    "Low Exploration": 0.1
}


def run_experiment(initial_epsilon):
    """
    Train a Q-learning agent using a specific
    initial exploration rate.
    """

    env = GridWorld()

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=initial_epsilon
    )

    episode_rewards = []
    episode_steps = []
    successes = []

    for episode in range(NUM_EPISODES):

        state = env.reset()

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        successes.append(done)

        # Gradually reduce exploration
        agent.epsilon = max(
            0.05,
            agent.epsilon * 0.995
        )

    return {
        "rewards": episode_rewards,
        "steps": episode_steps,
        "successes": successes
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = {}


for strategy_name, initial_epsilon in EXPLORATION_STRATEGIES.items():

    print(f"\nRunning: {strategy_name}")
    print(f"Initial epsilon: {initial_epsilon}")

    results[strategy_name] = run_experiment(
        initial_epsilon
    )

    data = results[strategy_name]

    successes = data["successes"]
    rewards = data["rewards"]
    steps = data["steps"]

    print(
        f"Final success rate: "
        f"{np.mean(successes[-WINDOW_SIZE:]):.2%}"
    )

    print(
        f"Final average reward: "
        f"{np.mean(rewards[-WINDOW_SIZE:]):.2f}"
    )

    print(
        f"Final average steps: "
        f"{np.mean(steps[-WINDOW_SIZE:]):.2f}"
    )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

with open(
    "experiments/exploration_results.pkl",
    "wb"
) as file:

    pickle.dump(results, file)


print(
    "\nRaw experiment results saved to "
    "experiments/exploration_results.pkl"
)


# ============================================================
# CALCULATE ROLLING STATISTICS
# ============================================================

success_rate_results = {}
reward_results = {}
step_results = {}


for strategy_name, data in results.items():

    successes = data["successes"]
    rewards = data["rewards"]
    steps = data["steps"]

    success_rates = []
    average_rewards = []
    average_steps = []

    for i in range(
        WINDOW_SIZE,
        NUM_EPISODES + 1
    ):

        success_rates.append(
            np.mean(
                successes[
                    i - WINDOW_SIZE:i
                ]
            )
        )

        average_rewards.append(
            np.mean(
                rewards[
                    i - WINDOW_SIZE:i
                ]
            )
        )

        average_steps.append(
            np.mean(
                steps[
                    i - WINDOW_SIZE:i
                ]
            )
        )

    success_rate_results[strategy_name] = success_rates
    reward_results[strategy_name] = average_rewards
    step_results[strategy_name] = average_steps


episodes = range(
    WINDOW_SIZE,
    NUM_EPISODES + 1
)


# ============================================================
# PLOT SUCCESS RATE
# ============================================================

plt.figure(figsize=(10, 5))

for strategy_name in EXPLORATION_STRATEGIES:

    plt.plot(
        episodes,
        success_rate_results[strategy_name],
        label=strategy_name
    )

plt.xlabel("Episode")
plt.ylabel("Success Rate")

plt.title(
    "Effect of Exploration on Learning Success Rate"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/exploration_success_rate.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT AVERAGE REWARD
# ============================================================

plt.figure(figsize=(10, 5))

for strategy_name in EXPLORATION_STRATEGIES:

    plt.plot(
        episodes,
        reward_results[strategy_name],
        label=strategy_name
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "Effect of Exploration on Average Reward"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/exploration_reward.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT AVERAGE STEPS
# ============================================================

plt.figure(figsize=(10, 5))

for strategy_name in EXPLORATION_STRATEGIES:

    plt.plot(
        episodes,
        step_results[strategy_name],
        label=strategy_name
    )

plt.xlabel("Episode")
plt.ylabel("Average Steps")

plt.title(
    "Effect of Exploration on Navigation Efficiency"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/exploration_steps.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\nExperiment complete.")

print("Saved:")
print("  experiments/exploration_results.pkl")
print("  plots/exploration_success_rate.png")
print("  plots/exploration_reward.png")
print("  plots/exploration_steps.png")

