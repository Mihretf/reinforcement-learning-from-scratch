import numpy as np
import matplotlib.pyplot as plt

from environment.grid_world import GridWorld
from scripts.q_learning import QLearningAgent


NUM_EPISODES = 1000
MAX_STEPS = 100


env = GridWorld()

agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=1.0
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

    if (episode + 1) % 100 == 0:
        success_rate = np.mean(successes[-100:])

        print(
            f"Episode {episode + 1}/{NUM_EPISODES} | "
            f"Reward: {total_reward:.1f} | "
            f"Steps: {steps} | "
            f"Success rate: {success_rate:.2%} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )


# Save learned Q-table
np.save(
    "experiments/q_table.npy",
    agent.q_table
)


# Calculate success rate over rolling 100-episode windows
window_size = 100

success_rates = []

for i in range(window_size, NUM_EPISODES + 1):
    rate = np.mean(successes[i - window_size:i])
    success_rates.append(rate)


episodes = range(window_size, NUM_EPISODES + 1)


# Plot 1: Episode rewards
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, NUM_EPISODES + 1),
    episode_rewards
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Reward")

plt.tight_layout()

plt.savefig(
    "plots/q_learning_rewards.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# Plot 2: Success rate
plt.figure(figsize=(10, 5))

plt.plot(
    episodes,
    success_rates
)

plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Q-Learning Success Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "plots/q_learning_success_rate.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\nQ-table saved to experiments/q_table.npy")
print("Reward plot saved to plots/q_learning_rewards.png")
print("Success-rate plot saved to plots/q_learning_success_rate.png")