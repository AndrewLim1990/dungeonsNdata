from collections import defaultdict
from utils.agent_utils import EGreedyPolicy
from utils.agent_utils import Experience
from utils.agent_utils import filter_illegal_actions
from utils.agent_utils import mean_sq_error
from utils.agent_utils import Memory

import copy
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


class DQN:
    def __init__(self, max_training_steps=5e6, epsilon_start=0.3, epsilon_end=0.05, alpha=1e-4,
                 gamma=0.999, update_frequency=5e4, memory_length=1024, batch_size=128):
        self.max_training_steps = max_training_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.alpha = alpha
        self.gamma = gamma
        self.update_frequency = update_frequency
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

        self.policy_net = None
        self.target_net = None
        self.memory = Memory(memory_length)
        self.name = "DQN"
        self.batch_size = batch_size

    @staticmethod
    def determine_enemy(creature, combat_handler):
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

    @staticmethod
    def determine_reward(damage_done=0):
        """
        :param creature:
        :param enemy:
        :param damage_done:
        :return:
        """
        reward = damage_done - 0.1
        return reward

    def initialize(self, creature, combat_handler):
        # Initialize weights if needed
        if self.policy_net is None:
            enemy = self.determine_enemy(creature=creature, combat_handler=combat_handler)
            state = combat_handler.get_current_state(creature=creature, enemy=enemy)
            self.initialize_weights(creature, state)

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)
        self.n_features = self.n_states + self.n_actions

        # Initialize weights
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, self.n_actions)
        )

        self.target_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, self.n_actions)
        )

        # Obtain position in feature list
        action_indicies = zip(creature.actions, range(self.n_actions))
        self.action_to_index = {action: index for action, index in action_indicies}
        self.index_to_action = {index: action for action, index in self.action_to_index.items()}
        return

    def get_best_action(self, state):
        """
        :param creature:
        :param state:
        :param report_q:
        :return:
        """
        q_vals = self.policy_net(state).detach()
        action_idx = q_vals.max(1)[1].view(1, 1)
        return action_idx

    def get_best_action_stochastic(self, state):
        """
        :param creature:
        :param state:
        :param report_q:
        :return:
        """
        p = self.policy_net(state).detach().numpy()
        p = 2**p  # np.exp(p)
        p = p / p.sum()
        action_idx = np.random.choice(range(p.shape[1]), p=p[0])
        action_idx = torch.tensor(action_idx).view(1, 1)
        return action_idx

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

        # Sample action indicies:
        eps_thresh = self.policy.get_epsilon(t=self.t)
        random_val = np.random.random()
        if random_val > eps_thresh:
            action_index = self.get_best_action(state)
        else:
            action_index = torch.tensor([[np.random.randint(self.n_actions)]], dtype=torch.long)
        self.t += 1

        q_val = self.policy_net(state)[0][action_index]

        # Returns either action index or action
        if return_index:
            return action_index, q_val
        else:
            return self.index_to_action[action_index.data.tolist()[0][0]], q_val

    def update_weights(self, predicted_batch, target_batch):
        loss = mean_sq_error(target=target_batch, predicted=predicted_batch)
        self.policy_net.zero_grad()
        loss.backward()

        # Update weights
        with torch.no_grad():
            for param in self.policy_net.parameters():
                param -= self.alpha * param.grad

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
        predicted_batch = self.policy_net(state_batch).gather(1, action_batch)
        self.update_weights(predicted_batch=predicted_batch, target_batch=target_batch)

    def calc_error(self, current_state, action_index, reward, next_state):
        predicted = self.policy_net(current_state).detach().gather(1, action_index)
        target = self.gamma * self.target_net(next_state).max(1)[0].detach().view(-1, 1) + reward
        priority = torch.abs(predicted - target).pow(0.7)
        return priority

    def update_step(self, action, creature, current_state, next_state, combat_handler):
        current_state = torch.from_numpy(current_state).float()
        action_index = torch.tensor([[self.action_to_index[action]]])
        enemy = self.determine_enemy(creature, combat_handler)
        damage_done = 0
        if next_state is not None:
            damage_done = (current_state - next_state)[0][1] * enemy.max_hit_points

        # Obtain reward
        reward = torch.tensor([[self.determine_reward(damage_done=damage_done)]]).float()

        # Add to experience replay
        self.memory.add(Experience(current_state, action_index, reward, next_state))

        # Update weights:
        if len(self.memory) >= self.memory.memory_length:
            self.learn_from_replay()

        # Update target network weights
        if (self.training_iteration + 1) % self.update_frequency == 0:
            print("w_stored <- w")
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.training_iteration += 1


class DoubleDQN(DQN):
    def __init__(self, max_training_steps=1e6, epsilon_start=0.3, epsilon_end=0.05, alpha=5e-2,
                 gamma=0.99, update_frequency=30000, memory_length=4096, batch_size=128):
        super().__init__(
            max_training_steps, epsilon_start, epsilon_end, alpha, gamma, update_frequency, memory_length, batch_size
        )
        self.name = "double_DQN"

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)
        self.n_features = self.n_states + self.n_actions

        h1 = self.n_actions

        # Initialize weights
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, self.n_actions),
        )
        self.target_net = copy.deepcopy(self.policy_net)

        # Obtain position in feature list
        action_indicies = zip(creature.actions, range(self.n_actions))
        self.action_to_index = {action: index for action, index in action_indicies}
        self.index_to_action = {index: action for action, index in self.action_to_index.items()}
        return

    def calc_error(self, current_state, action_index, reward, next_state):
        if next_state is None:
            evaluation = reward
        else:
            next_state = torch.tensor(next_state).float()
            next_action = self.policy_net(next_state).detach().max(1)[1].view(-1, 1)
            evaluation = self.target_net(next_state).detach().gather(1, next_action)
        target = self.gamma * evaluation + reward
        predicted = self.policy_net(current_state).detach().gather(1, action_index)
        error = predicted - target

        return error

    def learn_from_replay(self):
        # Sample experiences from memory
        batch = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*batch))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        evaluation_batch = torch.zeros((self.batch_size, 1))

        if non_final_mask.sum() >= 1:
            non_final_next_states = torch.cat([torch.tensor(s).float() for s in batch.next_state if s is not None])
            selected_actions = self.policy_net(non_final_next_states).max(1)[1].view(-1, 1)
            non_final_evaluation_batch = self.target_net(non_final_next_states).detach().gather(1, selected_actions)
            evaluation_batch[non_final_mask] = non_final_evaluation_batch

        # Calculate gradients
        target_batch = self.gamma * evaluation_batch + reward_batch
        predicted_batch = self.policy_net(state_batch).gather(1, action_batch)
        # q_before = self.policy_net(state_batch).detach()
        self.update_weights(predicted_batch=predicted_batch, target_batch=target_batch)
        # q_after = self.policy_net(state_batch).detach()
        # action_name = self.index_to_action[action_batch.numpy()[0][0]].name

        # print("Action: {}".format(action_name))
        # print("State: {}".format(state_batch))
        # print("Next State:{}".format(batch.next_state))
        # print("Reward: {}".format(reward_batch))
        # print("Predicted: {}".format(actual_batch))
        # print("Target: {}".format(target_batch))
        # print("Q Before: {}".format(q_before))
        # print("Q After: {}".format(q_after))
        # print("Q Delta: {}\n".format(q_after - q_before))
