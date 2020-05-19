from collections import defaultdict
from utils.agent_utils import EGreedyPolicy
from utils.agent_utils import Experience
from utils.agent_utils import filter_illegal_actions
from utils.agent_utils import mean_sq_error
from utils.agent_utils import Memory
from utils.agent_utils import PrioritizedMemory
from utils.agent_utils import SARSAExperience

import copy
import numpy as np
import torch

TIME_LIMIT = 1500
ROUND_ACTION_LIMIT = 50


class Strategy:
    def __init__(self):
        self.action_to_index = dict()
        self.index_to_action = dict()
        self.n_actions = None

    def update_step(self, *args, **kwargs):
        pass

    def update_strategy(self):
        pass

    def determine_reward(self, *args, **wargs):
        pass

    @staticmethod
    def determine_enemy(creature, combat_handler):
        enemy = None
        combatants = combat_handler.combatants
        for combatant in combatants:
            if combatant != creature:
                enemy = combatant
        return enemy

    def initialize(self, creature, combat_handler):
        # Obtain dictionaries translating index to actions and vice versa
        self.n_actions = len(creature.actions)
        action_indicies = zip(creature.actions, range(self.n_actions))
        self.action_to_index = {action: index for action, index in action_indicies}
        self.index_to_action = {index: action for action, index in self.action_to_index.items()}


class RandomStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "random"

    def sample_action(self, creature, combat_handler):
        actions = filter_illegal_actions(creature=creature, actions=creature.actions)
        action = np.random.choice(actions)
        return action

    @staticmethod
    def get_current_state(*args, **kwargs):
        return None

    @staticmethod
    def get_raw_state(*args, **kwargs):
        return [None]


class RangeAggression(Strategy):
    def __init__(self, *args, **kwargs):
        self.name = "ranged_aggression"

    def sample_action(self, creature, combat_handler):
        """
        Always uses "Arrow Shot"
        :param creature:
        :param combat_handler:
        :return:
        """
        actions = [creature.get_action("Arrow Shot"), creature.get_action("end_turn")]

        action = np.random.choice(actions, p=[0.95, 0.05])

        return action


class QLearningTabularAgent(Strategy):
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

    @staticmethod
    def get_current_state(creature, combat_handler):
        """
        Todo: Implement this.
        :param self:
        :return:
        """
        return 1

    def sample_action(self, creature, combat_handler):
        """
        :param creature:
        :param combat_handler:
        :return: action
        """
        actions = creature.actions
        actions = filter_illegal_actions(creature, actions)
        state = tuple(self.get_current_state(creature=creature, combat_handler=combat_handler)[0])
        best_action = self.get_best_action(creature, state)

        # Obtain action via e-greedy policy
        # Todo: Move self.t into policy
        # Todo: Rename "policy" to something better
        action = self.policy.sample_policy_action(actions, best_action, self.t)

        self.t += 1
        return action

    @staticmethod
    def determine_reward(creature, enemy):
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


class FunctionApproximation(Strategy):
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
        self.optimizer = None
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
    def get_raw_state(creature, enemy, combat_handler):
        raw_state = np.array([[
            creature.hit_points / creature.max_hit_points,                          # creature health
            enemy.hit_points / enemy.max_hit_points,                                # enemy health
            creature.location[0] / combat_handler.environment.room_width,           # creature x loc
            creature.location[1] / combat_handler.environment.room_length,          # creature y loc
            enemy.location[0] / combat_handler.environment.room_width,              # enemy x loc
            enemy.location[1] / combat_handler.environment.room_length,             # enemy y loc
            creature.attacks_used,                                                  # attacks used
            creature.movement_remaining / creature.speed,                           # remaining movement\
            (2 * creature.action_count - TIME_LIMIT) / TIME_LIMIT                   # num actions taken
        ]])
        return raw_state

    def get_current_state(self, creature, combat_handler):
        enemy = self.determine_enemy(creature, combat_handler)
        is_exceeded_time_limit = creature.action_count >= TIME_LIMIT

        if not(creature.is_alive()) or not(enemy.is_alive()) or is_exceeded_time_limit:
            current_state = None
        else:
            current_state = self.get_raw_state(creature, enemy, combat_handler)

        return current_state

    def initialize(self, creature, combat_handler):
        # Initialize weights if needed
        if self.policy_net is None:
            state = self.get_current_state(creature=creature, combat_handler=combat_handler)
            self.initialize_weights(creature, state)

        # Obtain dictionaries translating index to actions and vice versa
        action_indicies = zip(creature.actions, range(self.n_actions))
        self.action_to_index = {action: index for action, index in action_indicies}
        self.index_to_action = {index: action for action, index in self.action_to_index.items()}

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)
        self.n_features = self.n_states + self.n_actions

        # Initialize weights
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, self.n_actions)
        )
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())

    def get_best_action(self, state):
        """
        :param state:
        :return:
        """
        q_vals = self.policy_net(state).detach()
        action_idx = q_vals.max(1)[1].view(1, 1)
        return action_idx

    def sample_action(self, creature, combat_handler, increment_counter=True, state=None):
        """
        Returns an action or an action_index

        :param creature:
        :param combat_handler:
        :param increment_counter:
        :param state:
        :return: action_index
        """
        # Obtain state / actions:
        if state is None:
            state = torch.from_numpy(self.get_current_state(creature=creature, combat_handler=combat_handler)).float()

        # Sample action indicies:
        eps_thresh = self.policy.get_epsilon(t=self.t)
        random_val = np.random.random()
        if random_val > eps_thresh:
            action_index = self.get_best_action(state)
        else:
            action_index = torch.tensor([[np.random.randint(self.n_actions)]], dtype=torch.long)

        if increment_counter:
            self.t += 1

        # Return action
        return self.index_to_action[action_index.data.tolist()[0][0]]

    def update_weights(self, predicted_batch, target_batch):
        loss = mean_sq_error(target=target_batch, predicted=predicted_batch)
        self.policy_net.zero_grad()
        loss.backward()

        # Update weights
        with torch.no_grad():
            for param in self.policy_net.parameters():
                param -= self.alpha * param.grad

        return loss

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

    def update_step(self, action, creature, current_state, next_state, combat_handler):
        current_state = torch.from_numpy(current_state).float()
        action_index = torch.tensor([[self.action_to_index[action]]])

        # Obtain reward
        reward = torch.tensor([[self.determine_reward(creature, current_state, next_state, combat_handler)]]).float()

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

        return round(float(reward), 3)

    def determine_reward(self, creature, current_state, next_state, combat_handler):
        """
        :param creature:
        :param current_state:
        :param next_state:
        :param combat_handler:
        :return:
        """
        reward = 0

        enemy = self.determine_enemy(creature, combat_handler)
        raw_next_state = self.get_raw_state(creature, enemy, combat_handler)
        damage_done = (current_state - raw_next_state)[0][1]
        # damage_taken = (current_state - raw_next_state)[0][0] / 8
        # reward = round(float(damage_done) - float(damage_taken), 2) * 100
        reward += round(float(damage_done), 2) * 100

        return reward


class DoubleDQN(FunctionApproximation):
    def __init__(self, max_training_steps=1e6, epsilon_start=0.3, epsilon_end=0.05, alpha=1e-3,
                 gamma=0.99, update_frequency=30000, memory_length=4096, batch_size=128):
        super().__init__(
            max_training_steps, epsilon_start, epsilon_end, alpha, gamma, update_frequency, memory_length, batch_size
        )
        self.name = "double_DQN"
        self.optimizer = None

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)
        self.n_features = self.n_states + self.n_actions

        h1 = self.n_actions

        # Initialize weights
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, self.n_actions),
        )
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())

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
        self.update_weights(predicted_batch=predicted_batch, target_batch=target_batch)


class SARSA(FunctionApproximation):
    """
    SARSA
    """
    def __init__(self, max_training_steps=5e6, epsilon_start=0.5, epsilon_end=0.05, alpha=1e-4,
                 gamma=0.999, update_frequency=5e4, memory_length=1024, batch_size=128):
        super().__init__(
            max_training_steps, epsilon_start, epsilon_end, alpha, gamma, update_frequency, memory_length, batch_size
        )
        self.memory = PrioritizedMemory(memory_length, experience_type=SARSAExperience)
        self.name = 'SARSA'

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)

        h1 = self.n_actions

        # Initialize weights
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, self.n_actions),
        )

    def update_step(self, action, creature, current_state, next_state, combat_handler):
        current_state = torch.from_numpy(current_state).float()
        next_state = torch.from_numpy(next_state).float() if next_state is not None else None
        action_index = torch.tensor([[self.action_to_index[action]]])

        # Obtain reward
        reward = torch.tensor([[self.determine_reward(creature, current_state, next_state, combat_handler)]]).float()

        # Obtain next action
        next_action_index = None
        if next_state is not None:
            next_action = self.sample_action(
                creature=creature,
                combat_handler=combat_handler,
                increment_counter=False,
                state=next_state
            )
            next_action_index = creature.strategy.action_to_index[next_action]
            next_action_index = torch.tensor([[next_action_index]])

        # Add to experience replay
        self.memory.add(SARSAExperience(current_state, action_index, reward, next_state, next_action_index))

        # Update weights:
        if len(self.memory) >= self.memory.memory_length:
            self.learn_from_replay()

        return round(float(reward), 3)

    def learn_from_replay(self):
        # Sample experiences from memory
        batch, indicies = self.memory.sample(self.batch_size)
        batch = SARSAExperience(*zip(*batch))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        evaluation_batch = torch.zeros((self.batch_size, 1))

        # Todo: reuse non_final_mask for selection below
        # Todo: try to convert to tensor earlier when added to memory
        if non_final_mask.sum() >= 1:
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            non_final_next_actions = torch.tensor([a for a in batch.next_action if a is not None]).view(-1, 1)
            non_final_evaluation_batch = self.policy_net(non_final_next_states).gather(1, non_final_next_actions)
            evaluation_batch[non_final_mask] = non_final_evaluation_batch

        # Calculate gradients
        target_batch = self.gamma * evaluation_batch + reward_batch
        predicted_batch = self.policy_net(state_batch).gather(1, action_batch)
        self.update_weights(predicted_batch=predicted_batch, target_batch=target_batch)

        # Update priorities
        priorities = ((predicted_batch - target_batch) ** 2 + self.memory.epsilon) ** self.memory.alpha
        self.memory.update_priorities(indicies=indicies, priorities=priorities)

    def determine_reward(self, creature, current_state, next_state, combat_handler):
        """
        :param creature:
        :param current_state:
        :param next_state:
        :param combat_handler:
        :return:
        """
        reward = 0
        if next_state is None:
            is_dead = creature.hit_points < 0
            is_too_many_combat_actions = creature.action_count >= TIME_LIMIT
            if not(is_dead or is_too_many_combat_actions):
                reward = 100

        return reward
