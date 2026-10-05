import numpy as np

from environment.grid_world import GridWorld


NUM_EPISODES = 1000
MAX_STEPS = 100


env = GridWorld()

episode_rewards = []
episode_steps = []
successes = []


for episode in range(NUM_EPISODES):

    state = env.reset()

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < MAX_STEPS:

        # Choose an action completely at random
        action = np.random.randint(4)

        next_state, reward, done = env.step(action)

        state = next_state
        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    successes.append(done)


success_rate = np.mean(successes)
average_reward = np.mean(episode_rewards)
average_steps = np.mean(episode_steps)


print("Random Agent Baseline")
print("---------------------")
print(f"Episodes: {NUM_EPISODES}")
print(f"Success rate: {success_rate:.2%}")
print(f"Average reward: {average_reward:.2f}")
print(f"Average steps: {average_steps:.2f}")