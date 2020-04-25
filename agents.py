from collections import defaultdict
from utils.agent_utils import EGreedyPolicy
from utils.agent_utils import Experience
from utils.agent_utils import filter_illegal_actions
from utils.agent_utils import Memory

import numpy as np
import torch


class Agent:
    def __init__(self):
        pass


class TabularAgent(Agent):
    pass


class QLearningTabularAgent(TabularAgent):
    def __init__(self, max_training_steps=2e8, epsilon_start=0.01, epsilon_end=0.001, alpha=1e-2,
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
        self.name = "q_tabular"

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
        state = tuple(state[0])
        best_action_indicies = np.argsort(-self.Q[state])
        best_actions = [self.index_to_action[idx] for idx in best_action_indicies]

        # Filter out illegal actions
        best_actions = filter_illegal_actions(creature, best_actions)

        # Take best action amongst remaining actions
        best_action = best_actions[0]

        return best_action

    def sample_action(self, creature, combat_handler):
        """
        :param creature:
        :param combat_handler:
        :return: action
        """
        actions = creature.actions
        actions = filter_illegal_actions(creature, actions)
        enemy = creature.player.strategy.determine_enemy(creature, combat_handler=combat_handler)
        state = tuple(combat_handler.get_current_state(creature, enemy)[0])
        best_action = self.get_best_action(creature, state)

        # Obtain action via e-greedy policy
        action = self.policy.sample_policy_action(actions, best_action, self.t)

        self.t += 1
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
        current_state = tuple(current_state)
        next_state = tuple(next_state)
        # Perform action, obtain s', r
        enemy = self.determine_enemy(creature, combat_handler)
        reward = self.determine_reward(creature, enemy)

        # Perform update:
        action_index = self.action_to_index[action]
        diff = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[current_state][action_index]
        self.Q[current_state][action_index] += self.alpha * diff
        return


class FunctionApproximation:
    def __init__(self, max_training_steps=2e7, epsilon_start=0.1, epsilon_end=0.005, alpha=1e-1,
                 gamma=0.99, update_frequency=5e5):
        self.max_training_steps = max_training_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.alpha = alpha
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.name = "linear_function_approximation"
        self.training_iteration = 0
        self.t = 0
        self.policy = EGreedyPolicy(n_steps=max_training_steps, epsilon_start=epsilon_start, epsilon_end=epsilon_end)
        self.w = None
        self.w_stored = None
        self.action_to_index = None
        self.index_to_action = None
        self.n_states = None
        self.n_actions = None
        self.n_features = None

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

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)
        self.n_features = self.n_states + self.n_actions

        # Initialize weights
        self.w = np.random.random((self.n_states, self.n_actions)) * 1e-3
        self.w_stored = self.w

        # Obtain position in feature list
        action_indicies = zip(creature.actions, range(self.n_actions))
        self.action_to_index = {action: index for action, index in action_indicies}
        self.index_to_action = {index: action for action, index in self.action_to_index.items()}

    def get_best_action(self, creature, state, report_q=False):
        if self.w is None:
            self.initialize_weights(creature, state)

        # Calc q-values per action
        q_values = np.dot(state, self.w, )

        # Get legal actions
        legal_actions = filter_illegal_actions(creature, creature.actions)
        legal_action_indicies = [self.action_to_index[action] for action in legal_actions]

        # Select best legal action
        action_indicies = np.argsort(-q_values)[0]
        action_index = [idx for idx in action_indicies if idx in legal_action_indicies][0]
        action = self.index_to_action[action_index]

        if report_q:
            return action, np.max(q_values)
        else:
            return action

    def sample_action(self, creature, combat_handler):
        """
        :param creature:
        :param combat_handler:
        :return: action
        """
        actions = creature.actions
        actions = filter_illegal_actions(creature, actions)
        enemy = creature.player.strategy.determine_enemy(creature, combat_handler=combat_handler)
        state = combat_handler.get_current_state(creature, enemy)
        best_action = self.get_best_action(creature, state)

        # Obtain action via e-greedy policy
        action = self.policy.sample_policy_action(actions, best_action, self.t)

        self.t += 1
        return action

    def determine_reward(self, creature, enemy):
        """
        :param creature:
        :param enemy:
        :return:
        """
        reward = -0.01
        if enemy.is_alive() is False:
            reward = 100
        elif creature.is_alive() is False:
            reward = -100
        return reward

    def best_old_q(self, state):
        # Calc q-values per action
        q_values = np.dot(state, self.w_stored)

        # Select best action
        q_val = np.max(q_values)
        return q_val

    def construct_input_features(self, action, state):
        one_hot_action = np.zeros(self.n_actions)
        action_index = self.action_to_index[action]
        one_hot_action[action_index] = 1
        one_hot_action = one_hot_action.reshape(1, -1)
        input_features = np.hstack((one_hot_action, state))

        return input_features

    def calc_q_vals(self, state, w):
        q_val = np.dot(state, w)

        return q_val

    def update_step(self, action, creature, current_state, next_state, combat_handler):
        self.training_iteration += 1
        enemy = self.determine_enemy(creature, combat_handler)
        reward = self.determine_reward(creature, enemy)
        target = reward + self.gamma * self.best_old_q(state=next_state)
        actual = self.calc_q_vals(current_state, self.w)

        action_mask = np.zeros(len(creature.actions))
        action_idx = self.action_to_index[action]
        action_mask[action_idx] = 1
        error = target - actual
        error = error * action_mask
        diff = np.dot(error.T, current_state).T

        # print("Action: {} (reward: {}) (actual: {} -> target: {})".format(action.name, reward,  actual[0][action_idx], target))
        # print("current_state: {}".format(current_state))
        # print("next_state:    {}".format(next_state))

        q_before = self.calc_q_vals(current_state, self.w)
        self.w += self.alpha * diff
        q_after = self.calc_q_vals(current_state, self.w)
        q_diff = q_after - q_before
        # print("q_before: {}".format(q_before))
        # print("q_after:  {}".format(q_after))
        # print("q_diff:   {}\n".format(q_diff))

        if self.training_iteration % self.update_frequency == 0:
            print("w_stored <- w")
            self.w_stored = self.w

    def initialize(self, *args, **kwargs):
        pass


class DQN(FunctionApproximation):
    def __init__(self, max_training_steps=2e7, epsilon_start=0.9, epsilon_end=0.05, alpha=1e-3,
                 gamma=0.99, update_frequency=5e5, memory_length=10000, batch_size=64):
        super().__init__(max_training_steps, epsilon_start, epsilon_end, alpha, gamma, update_frequency)
        self.policy_net = None
        self.target_net = None
        self.memory = Memory(memory_length)
        self.name = "DQN"
        self.batch_size = batch_size

    def loss_fn(self, target, predicted):
        sq_error = (predicted - target).pow(2)
        loss = torch.mean(sq_error)

        return loss

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)
        self.n_features = self.n_states + self.n_actions

        h1 = self.n_states
        h2 = self.n_states

        # Initialize weights
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, self.n_actions)
        )

        self.target_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, self.n_actions)
        )

        # Obtain position in feature list
        action_indicies = zip(creature.actions, range(self.n_actions))
        self.action_to_index = {action: index for action, index in action_indicies}
        self.index_to_action = {index: action for action, index in self.action_to_index.items()}

    def get_best_action(self, creature, state, report_q=False):
        """
        :param creature:
        :param state:
        :param report_q:
        :return:
        """
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def sample_action(self, creature, combat_handler, return_index=False):
        """
        Returns an action or an action_index

        :param creature:
        :param combat_handler:
        :param return_index:
        :return: action_index
        """
        # Obtain state / actions:
        enemy = creature.player.strategy.determine_enemy(creature, combat_handler=combat_handler)
        state = torch.from_numpy(combat_handler.get_current_state(creature, enemy)).float()

        # Initialize weights if needed
        if self.policy_net is None:
            self.initialize_weights(creature, state)

        # Sample action indicies:
        eps_thresh = self.policy.get_epsilon(t=self.t)
        random_val = np.random.random()
        if random_val > eps_thresh:
            action_index = self.get_best_action(creature, state)
        else:
            action_index = torch.tensor([[np.random.randint(self.n_actions)]], dtype=torch.long)
        self.t += 1

        # Returns either action index or action
        if return_index:
            return action_index
        else:
            return self.index_to_action[action_index.data.tolist()[0][0]]

    def learn_from_replay(self):
        # Sample experiences from memory
        batch = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*batch))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Calculate gradients
        target_batch = self.gamma * self.target_net(next_state_batch).max(1)[0].detach().view(-1, 1) + reward_batch
        actual_batch = self.policy_net(state_batch).gather(1, action_batch)
        loss = self.loss_fn(target=target_batch, predicted=actual_batch)
        self.policy_net.zero_grad()
        loss.backward()

        # Update weights
        with torch.no_grad():
            for param in self.policy_net.parameters():
                param -= self.alpha * param.grad

    def update_step(self, action, creature, current_state, next_state, combat_handler):
        current_state = torch.from_numpy(current_state).float()
        next_state = torch.from_numpy(next_state).float()
        action_index = torch.tensor([[self.action_to_index[action]]])
        enemy = self.determine_enemy(creature, combat_handler)
        reward = torch.tensor([[self.determine_reward(creature, enemy)]]).float()

        # Add to experience replay
        self.memory.add((current_state, action_index, reward, next_state))

        # Update weights:
        if len(self.memory) >= self.memory.memory_length:
            self.learn_from_replay()

        # Update target network weights
        if (self.training_iteration + 1) % self.update_frequency == 0:
            print("w_stored <- w")
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.training_iteration += 1



