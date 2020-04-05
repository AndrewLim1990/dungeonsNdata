from collections import defaultdict

import numpy as np

from settings import SUCCESSFUL_PLAYER_TURN_SIGNAL
from utils.agent_utils import EGreedyPolicy


class Agent:
    def __init__(self):
        pass


class TabularAgent(Agent):
    pass


class QLearningTabularAgent(TabularAgent):
    def __init__(self, combat_handler, max_training_steps=1e5, epsilon_start=0.5, epsilon_end=0.05, alpha=1e-3,
                 gamma=0.95):
        super().__init__()
        self.max_training_steps = int(max_training_steps)
        self.combat_hander = combat_handler
        self.policy = EGreedyPolicy(n_steps=max_training_steps, epsilon_start=epsilon_start, epsilon_end=epsilon_end)
        self.alpha = alpha
        self.gamma = gamma

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

    def obtain_current_state(self, creature):
        """
        :param creature:
        :param enemy:
        :return:
        """
        enemy = self.determine_enemy(creature)
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
        state = self.obtain_current_state(creature)
        best_action = self.get_best_action(state)

        # Obtain action via e-greedy policy
        action = self.policy.sample_action(actions, best_action, t)

        return action

    def determine_reward(self, enemy):
        reward = 0
        if enemy.is_alive() is False:
            reward = 10
        return reward

    def train(self, creature, combat_handler):
        self.initialize_q(creature)
        for t in range(self.max_training_steps):
            # Sample an action a given s
            action = self.sample_action(t)

            # Perform action, obtain s', r
            enemy = self.determine_enemy(creature)
            current_state = self.obtain_current_state(creature)
            self.take_action(creature, action, enemy, combat_handler)
            next_state = self.obtain_current_state(creature)
            reward = self.determine_reward(enemy)

            # Perform update:
            self.Q[current_state] += self.alpha * (
                        reward + self.gamma * np.max(self.Q[next_state]) - self.Q[current_state])

    def take_action(self, creature, action, enemy, combat_handler):
        """
        :return: action
        """
        meta_data_list = list()

        starting_location = creature.location
        action_signal, meta_data = creature.use_action(
            action,
            combat_handler=combat_handler,
            target_creature=enemy
        )
        meta_data_list.append(meta_data)
        meta_data = {"starting_location": starting_location}
        meta_data_list.append(meta_data)

        return SUCCESSFUL_PLAYER_TURN_SIGNAL, meta_data_list
