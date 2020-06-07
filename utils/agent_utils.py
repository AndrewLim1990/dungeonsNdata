from actions import Attack
from actions import Move
from collections import Counter
from collections import namedtuple
from itertools import compress
from operator import itemgetter
from torch.distributions import Categorical

import numpy as np
import random
import torch

Experience = namedtuple("Experience", ("state", "action", "reward", "next_state"))
SARSAExperience = namedtuple("SARSAExperience", ("state", "action", "reward", "next_state", "next_action"))


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


class Memory:
    def __init__(self, memory_length, experience_type=Experience):
        self.memory_length = memory_length
        self.memory = list()
        self.idx = 0
        self.experience_type = experience_type

    def __len__(self):
        return len(self.memory)

    def add(self, experience):
        is_under_max_length = len(self.memory) < self.memory_length
        if is_under_max_length:
            self.memory.append(self.experience_type(*experience))
        else:
            self.idx = self.idx % self.memory_length
            self.memory[self.idx] = self.experience_type(*experience)
            self.idx += 1
            pass

    def sample(self, n):
        """
        Sample from memory with replacement
        :param n:
        :return:
        """
        memories = random.sample(self.memory, n)
        return memories


class PrioritizedMemory(Memory):
    def __init__(self, memory_length, experience_type=Experience, alpha=0.6, epsilon=1e-6, beta=0.4):
        super().__init__(memory_length, experience_type)
        self.priorities = np.zeros(self.memory_length, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def add(self, experience):
        max_priority = self.priorities.max() if self.memory else 1.0

        is_under_max_length = len(self.memory) < self.memory_length
        if is_under_max_length:
            self.memory.append(self.experience_type(*experience))
        else:
            self.idx = self.idx % self.memory_length
            self.memory[self.idx] = self.experience_type(*experience)

        self.priorities[self.idx] = max_priority
        self.idx += 1

    def sample(self, n):
        prob = self.priorities / self.priorities.sum()
        indicies = np.random.choice(self.memory_length, n, p=prob)
        memories = list(itemgetter(*indicies)(self.memory))

        weights = (self.memory_length * prob[indicies]) ** (-self.beta)
        weights /= weights
        weights = torch.tensor(weights).view(-1, 1)

        return memories, indicies, weights

    def update_priorities(self, indicies, priorities):
        self.priorities[indicies.tolist()] = priorities.detach().data.numpy().reshape(-1)


def mean_sq_error(target, predicted, emphasis_weights=None):
    sq_error = (predicted - target).pow(2)

    if emphasis_weights is not None:
        sq_error = sq_error * emphasis_weights

    loss = torch.mean(sq_error)

    return loss


def filter_out_final_states(batch_data, non_final_mask):
    non_final_next_states = list(batch_data)
    non_final_next_states = list(compress(non_final_next_states, non_final_mask.tolist()))

    return non_final_next_states


class DuelingNet(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_hidden_units):
        super(DuelingNet, self).__init__()
        self.layer_1 = torch.nn.Linear(n_features, n_hidden_units)

        self.value_layer_1 = torch.nn.Linear(n_hidden_units, n_hidden_units)
        self.value_layer_2 = torch.nn.Linear(n_hidden_units, 1)

        self.advantage_layer_1 = torch.nn.Linear(n_hidden_units, n_hidden_units)
        self.advantage_layer_2 = torch.nn.Linear(n_hidden_units, n_outputs)

        self.relu = torch.nn.ReLU()

    def forward(self, state):
        layer_1_output = self.layer_1(state)
        layer_1_output = self.relu(layer_1_output)

        value_output = self.value_layer_1(layer_1_output)
        value_output = self.relu(value_output)
        value_output = self.value_layer_2(value_output)

        advantage_output = self.advantage_layer_1(layer_1_output)
        advantage_output = self.relu(advantage_output)
        advantage_output = self.advantage_layer_2(advantage_output)

        q_output = value_output + advantage_output - advantage_output.mean(dim=1, keepdim=True)

        return q_output


class ActorCritic(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_hidden_units):
        """
        Todo: Research to see if Actor and Critic should share some layers
        Todo: Look into initialization
        :param n_features:
        :param n_outputs:
        :param n_hidden_units:
        """
        super(ActorCritic, self).__init__()

        # Actor
        self.actor_layer = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units, n_outputs),
            torch.nn.Softmax(dim=-1)
        )

        # Critic
        self.critic_layer = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units, 1)
        )

    def forward(self, state):
        actor_output = self.actor_layer(state)
        value = self.critic_layer(state)
        dist = Categorical(actor_output)

        return dist, value

    def evaluate(self, state, action_index):
        actor_output = self.actor_layer(state)
        dist = Categorical(actor_output)
        entropy = dist.entropy().mean()
        log_probs = dist.log_prob(action_index.view(-1))
        value = self.critic_layer(state)

        return log_probs, value, entropy
