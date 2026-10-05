from environment.grid_world import GridWorld


env = GridWorld()

state = env.reset()

print("Initial state:", state)
env.render()

next_state, reward, done = env.step(GridWorld.RIGHT)

print("After moving right:")
print("State:", next_state)
print("Reward:", reward)
print("Done:", done)

env.render()