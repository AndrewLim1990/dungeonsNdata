import numpy as np
from collections import Counter


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

    def sample_policy_action(self, actions, best_action, t):
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


def calc_win_percentage(winner_list, creatures):
    total_games_played = len(winner_list)
    counts = Counter(winner_list)
    win_percentages = {creature: num_wins/total_games_played for creature, num_wins in counts.items()}
    for creature in creatures:
        if creature.name not in win_percentages.keys():
            win_percentages[creature.name] = 0
    win_percentages = sorted(win_percentages.items(), key=lambda x: x[0])
    return win_percentages


# def average_action_q(Q):
#     states = Q.keys()
#     tracker = defaultdict(list)
#     for state in states:
#         actions = Q[state].keys()
#         for action in actions:
#             tracker[action].append(Q[state][action])
