from actions import Attack
from actions import Move
from collections import Counter
from collections import namedtuple

import numpy as np
import random
import torch


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


def classlookup(cls):
    c = list(cls.__bases__)
    for base in c:
        c.extend(classlookup(base))
    return c


def filter_illegal_actions(creature, actions):
    """
    :param creature:
    :param actions:
    :return actions:
    """
    # Filter out illegal moves
    has_movement = creature.movement_remaining > 0
    if not has_movement:
        actions = [action for action in actions if Move not in classlookup(type(action)) + [type(action)]]

    # Filter out illegal attacks
    has_attack = creature.attacks_used < creature.attacks_allowed
    if not has_attack:
        actions = [action for action in actions if Attack not in classlookup(type(action)) + [type(action)]]

    return actions


Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "priority"))


class Memory:
    def __init__(self, memory_length):
        self.memory_length = memory_length
        self.memory = list()
        self.idx = 0

    def __len__(self):
        return len(self.memory)

    def add(self, experience):
        is_under_max_length = len(self.memory) < self.memory_length
        if is_under_max_length:
            self.memory.append(Experience(*experience))
        else:
            self.idx = self.idx % self.memory_length
            self.memory[self.idx] = experience

    def sample(self, n):
        """
        Sample from memory with replacement
        :param n:
        :return:
        """
        all_exp = Experience(*zip(*self.memory))
        priorities = np.array(list(all_exp.priority))
        priorities = priorities / priorities.sum()
        memory_indicies = np.random.choice(range(len(self.memory)), n, p=priorities, replace=False)
        memories = [self.memory[idx] for idx in memory_indicies]
        # memories = random.sample(self.memory, n)
        return memories
