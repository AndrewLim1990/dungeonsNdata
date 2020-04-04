import numpy as np


class Agent:
    def __init__(self):
        pass


class TabularAgent(Agent):
    pass


class QLearningTabularAgent(TabularAgent):
    def __init__(self, max_training_steps=1e5):
        super().__init__()
        self.max_training_steps = int(max_training_steps)

    def train(self):
        for i in range(self.max_training_steps):
            pass
            # Sample an action a given s
            # Perform action, obtain s', r
            # Record s, a, r, s' in replay buffer

    def perform_action(self):
        """

        :return: action
        """
        action = None
        return action
