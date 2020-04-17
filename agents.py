from actions import Attack
from actions import Move
from collections import defaultdict
from utils. agent_utils import classlookup
from utils.agent_utils import EGreedyPolicy

import numpy as np


class Agent:
    def __init__(self):
        pass


class TabularAgent(Agent):
    pass


class QLearningTabularAgent(TabularAgent):
    def __init__(self, max_training_steps=2e7, epsilon_start=0.01, epsilon_end=0.0005, alpha=1e-1,
                 gamma=0.999):
        """
        :param max_training_steps:
        :param epsilon_start:
        :param epsilon_end:
        :param alpha:
        :param gamma:
        """
        super().__init__()
        self.max_training_steps = int(max_training_steps)
        self.policy = EGreedyPolicy(n_steps=max_training_steps, epsilon_start=epsilon_start, epsilon_end=epsilon_end)
        self.alpha = alpha
        self.gamma = gamma
        self.Q = None
        self.action_to_index = None
        self.index_to_action = None
        self.t = 0
        self.last_action = None
        self.incoming_reward = None

    def initialize_q(self, creature):
        """
        :param creature:
        :return:
        """
        num_actions = len(creature.actions)
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        self.action_to_index = {k: v for k, v in zip(creature.actions, range(num_actions))}
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}

    def initialize(self, creature):
        """
        :param creature:
        :return:
        """
        self.initialize_q(creature)

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

    def get_best_action(self, creature, state):
        """
        :param state:
        :param creature:
        :return:
        """
        # Get list of actions ordered by bestness
        best_action_indicies = np.argsort(-self.Q[state])
        saved_action = self.index_to_action[best_action_indicies[0]]
        best_actions = [self.index_to_action[idx] for idx in best_action_indicies]

        # Filter out illegal actions
        best_actions = self.filter_illegal_actions(creature, best_actions)

        # Take best action amongst remaining actions
        best_action = best_actions[0]

        # print("Actual best action: {}, Taken best action: {}".format(saved_action.name, best_action.name))

        return best_action

    def obtain_current_state(self, creature, combat_handler):
        """
        :param creature:
        :param enemy:
        :return:
        """
        enemy = self.determine_enemy(creature, combat_handler)
        current_state = (
            creature.location[0],  # creature x loc
            creature.location[1],  # creature y loc
            enemy.location[0],  # enemy x loc
            enemy.location[1],  # enemy y loc
        )
        return current_state

    def filter_illegal_actions(self, creature, actions):
        """
        :param creature:
        :param actions:
        :return:
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

    def sample_action(self, creature, combat_handler):
        """
        :param creature:
        :param combat_handler:
        :return: action
        """
        actions = creature.actions
        actions = self.filter_illegal_actions(creature, actions)
        enemy = creature.player.strategy.determine_enemy(creature, combat_handler=combat_handler)
        state = combat_handler.get_current_state(creature, enemy)
        best_action = self.get_best_action(creature, state)

        # Obtain action via e-greedy policy
        action = self.policy.sample_policy_action(actions, best_action, self.t)

        self.t += 1
        # print("----> Action: {} ({})".format(action.name, self.action_to_index[action]))
        return action

    def determine_reward(self, creature, enemy):
        """
        :param creature:
        :param enemy:
        :return:
        """
        reward = -0.0001
        if enemy.is_alive() is False:
            reward = 100
        elif creature.is_alive() is False:
            reward = -100
        return reward

    def update_step(self, action, creature, current_state, next_state, combat_handler):
        """
        :param action:
        :param creature:
        :param current_state:
        :param next_state:
        :param combat_handler:
        :return:
        """
        # Perform action, obtain s', r
        enemy = self.determine_enemy(creature, combat_handler)
        reward = self.determine_reward(creature, enemy)
        # print("REWARD: {}".format(reward))

        # Perform update:
        action_index = self.action_to_index[action]
        diff = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[current_state][action_index]
        # print("Diff: {}".format(diff))
        # print("BEFORE: {}: {}".format(current_state, self.Q[current_state]))
        self.Q[current_state][action_index] += self.alpha * diff
        # print("AFTER: {}: {}".format(current_state, self.Q[current_state]))
        # print("NEXT: {}: {}\n".format(next_state, self.Q[next_state]))
        return


class DQN:
    pass