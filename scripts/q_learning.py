import numpy as np

from environment.grid_world import GridWorld


class QLearningAgent:

    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # 64 states × 4 possible actions
        self.q_table = np.zeros((64, 4))

    def state_to_index(self, state):
        """
        Convert (row, column) into a single Q-table index.
        """

        row, col = state

        return row * 8 + col

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy exploration.
        """

        state_index = self.state_to_index(state)

        # Explore
        if np.random.random() < self.epsilon:
            return np.random.randint(4)

        # Exploit
        return np.argmax(self.q_table[state_index])

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value using the Q-learning rule.
        """

        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        current_q = self.q_table[state_index, action]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state_index])
            target = reward + self.discount_factor * best_next_q

        self.q_table[state_index, action] += (
            self.learning_rate * (target - current_q)
        )

