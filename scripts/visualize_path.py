import numpy as np
import matplotlib.pyplot as plt

from environment.grid_world import GridWorld
from scripts.q_learning import QLearningAgent


env = GridWorld()

agent = QLearningAgent()

agent.q_table = np.load("experiments/q_table.npy")
agent.epsilon = 0.0

state = env.reset()

path = [state]
done = False

for _ in range(100):
    action = agent.choose_action(state)

    next_state, reward, done = env.step(action)

    path.append(next_state)
    state = next_state

    if done:
        break


# Create grid
grid = np.zeros((env.rows, env.cols))

# Mark obstacles
for row, col in env.obstacles:
    grid[row, col] = -1


fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(grid, cmap="Greys", vmin=-1, vmax=1)

# Draw the learned path
path_rows = [position[0] for position in path]
path_cols = [position[1] for position in path]

ax.plot(
    path_cols,
    path_rows,
    marker="o",
    linewidth=3
)

# Start
ax.scatter(
    env.start[1],
    env.start[0],
    s=250,
    marker="o",
    label="Start"
)

# Goal
ax.scatter(
    env.goal[1],
    env.goal[0],
    s=300,
    marker="*",
    label="Goal"
)

# Label each step
for step, (row, col) in enumerate(path):
    ax.text(
        col,
        row,
        str(step),
        ha="center",
        va="center",
        fontsize=9
    )

ax.set_xticks(range(env.cols))
ax.set_yticks(range(env.rows))

ax.set_title("Learned Q-Learning Path")
ax.set_xlabel("Column")
ax.set_ylabel("Row")

ax.grid(True)

ax.legend()

plt.tight_layout()

plt.savefig(
    "plots/learned_path.png",
    dpi=150
)

plt.show()