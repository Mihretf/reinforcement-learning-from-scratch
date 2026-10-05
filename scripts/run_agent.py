import numpy as np

from environment.grid_world import GridWorld
from scripts.q_learning import QLearningAgent


env = GridWorld()

agent = QLearningAgent()

agent.q_table = np.load("experiments/q_table.npy")

# Disable exploration.
agent.epsilon = 0.0

state = env.reset()

path = [state]
total_reward = 0
done = False

for step in range(100):

    action = agent.choose_action(state)

    next_state, reward, done = env.step(action)

    path.append(next_state)
    total_reward += reward

    state = next_state

    if done:
        break


print("Learned path:")
print(path)

print("\nNumber of steps:", len(path) - 1)
print("Total reward:", total_reward)
print("Reached goal:", done)