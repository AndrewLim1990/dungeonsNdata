import numpy as np


class EGreedyPolicy:
    """
    Samples actions at e-greedy rate.
    """
    def __init__(self, n_steps, epsilon_start, epsilon_end):
        """
        :param n_steps:
        :param epsilon_start:
        :param epsilon_end:
        """
        self.n_steps = n_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

    def get_epsilon(self, t):
        """
        :param t:
        :return: epsilon
        """
        if t > self.n_steps:
            epsilon = self.epsilon_end
        else:
            epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) / self.n_steps * t
        return epsilon

    def sample_action(self, actions, best_action, t):
        """
        :param actions:
        :param best_action:
        :param t:
        :return:
        """
        epsilon = self.get_epsilon(t)
        should_explore = np.random.binomial(n=1, p=epsilon)
        if should_explore:
            action = np.random.choice(actions)
        else:
            action = best_action

        return action
