import numpy as np


class GridWorld:
    """
    8x8 grid-world environment for reinforcement learning.

    The agent starts at (0, 0) and must reach the goal at (7, 7).
    """

    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self):
        self.rows = 8
        self.cols = 8

        self.start = (0, 0)
        self.goal = (7, 7)

        # Fixed obstacle layout
        self.obstacles = {
            (1, 1), (1, 2), (1, 3),
            (2, 3),
            (3, 1), (3, 3), (3, 4), (3, 5),
            (4, 5),
            (5, 1), (5, 2), (5, 3), (5, 4),
            (6, 4),
        }

        self.position = self.start

    def reset(self):
        """
        Reset the environment to the starting position.

        Returns:
            tuple: Starting state (row, column)
        """
        self.position = self.start
        return self.position

    def step(self, action):
        """
        Take one action in the environment.

        Args:
            action (int): One of UP, DOWN, LEFT, RIGHT.

        Returns:
            next_state (tuple): New agent position.
            reward (int): Reward received.
            done (bool): Whether the episode has ended.
        """

        row, col = self.position

        # Calculate proposed new position
        if action == self.UP:
            new_position = (row - 1, col)

        elif action == self.DOWN:
            new_position = (row + 1, col)

        elif action == self.LEFT:
            new_position = (row, col - 1)

        elif action == self.RIGHT:
            new_position = (row, col + 1)

        else:
            raise ValueError("Invalid action.")

        # Check whether the move leaves the grid
        new_row, new_col = new_position

        if (
            new_row < 0
            or new_row >= self.rows
            or new_col < 0
            or new_col >= self.cols
        ):
            return self.position, -2, False

        # Check whether the move hits an obstacle
        if new_position in self.obstacles:
            return self.position, -2, False

        # Move the agent
        self.position = new_position

        # Check whether the goal was reached
        if self.position == self.goal:
            return self.position, 20, True

        # Normal movement
        return self.position, -1, False

    def render(self):
        """
        Print the current environment.
        """

        for row in range(self.rows):
            line = []

            for col in range(self.cols):
                position = (row, col)

                if position == self.position:
                    symbol = "A"

                elif position == self.goal:
                    symbol = "G"

                elif position in self.obstacles:
                    symbol = "#"

                else:
                    symbol = "."

                line.append(symbol)

            print(" ".join(line))

        print()