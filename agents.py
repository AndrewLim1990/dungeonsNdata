from collections import defaultdict
from utils.agent_utils import EGreedyPolicy

import numpy as np


class Agent:
    def __init__(self):
        pass


class TabularAgent(Agent):
    pass


class QLearningTabularAgent(TabularAgent):
    def __init__(self, combat_handler, max_training_steps=1e5, epsilon_start=0.5, epsilon_end=0.05):
        super().__init__()
        self.max_training_steps = int(max_training_steps)
        self.combat_hander = combat_handler
        self.policy = EGreedyPolicy(n_steps=max_training_steps, epsilon_start=epsilon_start, epsilon_end=epsilon_end)

    def initialize_q(self, creature):
        num_actions = len(creature.actions)
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        self.action_to_index = {k: v for k, v in zip(range(num_actions), creature.actions)}
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}

    def determine_enemy(self, creature, combat_handler):
        """
        :param creature:
        :param combat_handler:
        :return enemy:
        """
        enemy = None
        combatants = combat_handler.combatants
        for combatant in combatants:
            if combatant != creature:
                enemy = combatant
        return enemy

    def get_best_action(self, state):
        """
        :param state:
        :return:
        """
        best_action_index = np.argmax(self.Q[state])
        best_action = self.index_to_action[best_action_index]
        return best_action

    def obtain_current_state(self, creature, enemy):
        """
        :param creature:
        :param enemy:
        :return:
        """
        current_state = (
            creature.location[0],  # creature x loc
            creature.location[1],  # creature y loc
            creature.is_alive,  # creature is alive?
            enemy.location[0],  # enemy x loc
            enemy.location[1],  # enemy y loc
            enemy.is_alive  # enemy is alive?
        )
        return current_state

    def sample_action(self, t, creature):
        actions = creature.actions
        enemy = self.determine_enemy(creature)
        state = self.obtain_current_state(creature, enemy)
        best_action = self.get_best_action(state)

        # Obtain action via e-greedy policy
        action = self.policy.sample_action(actions, best_action, t)

        return action

    def train(self):
        for t in range(self.max_training_steps):
            # Sample an action a given s
            action = self.sample_action(t)
            # Perform action, obtain s', r
            # Record s, a, r, s' in replay buffer

    def perform_action(self):
        """
        :return: action
        """
        action = None
        return action
