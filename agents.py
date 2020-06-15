from collections import defaultdict
from utils.agent_utils import EGreedyPolicy
from utils.agent_utils import Experience
from utils.agent_utils import filter_illegal_actions
from utils.agent_utils import filter_out_final_states
from utils.agent_utils import mean_sq_error
from utils.agent_utils import PrioritizedMemory
from utils.agent_utils import SARSAExperience
from utils.agent_utils import DuelingNet
from utils.agent_utils import ActorCritic

import copy
import numpy as np
import torch

TIME_LIMIT = 1500
ROUND_ACTION_LIMIT = 50
VALUE_INDEX = -1
NUM_ATTACKS_USED = 6


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
        return 0

    def update_step_trajectory(self, *args, **wargs):
        pass

    def get_current_state(self, creature, combat_handler):
        enemy = self.determine_enemy(creature, combat_handler)
        is_exceeded_time_limit = creature.action_count >= TIME_LIMIT

        if not(creature.is_alive()) or not(enemy.is_alive()) or is_exceeded_time_limit:
            current_state = None
        else:
            current_state = self.get_raw_state(creature, enemy, combat_handler)

        return current_state

    @staticmethod
    def determine_enemy(creature, combat_handler):
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
            creature.movement_remaining / creature.speed,                           # remaining movement
            creature.action_count / TIME_LIMIT                                      # num actions taken
        ]])
        raw_state = torch.from_numpy(raw_state).float()
        return raw_state

    def initialize(self, creature, combat_handler):
        # Obtain dictionaries translating index to actions and vice versa
        self.n_actions = len(creature.actions)
        action_indicies = zip(creature.actions, range(self.n_actions))
        self.action_to_index = {action: index for action, index in action_indicies}
        self.index_to_action = {index: action for action, index in self.action_to_index.items()}

    def evaluate_state_and_action(self, *args, **kwargs):
        return None, None


class RandomStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "random"

    def sample_action(self, creature, combat_handler):
        actions = filter_illegal_actions(creature=creature, actions=creature.actions)
        action = np.random.choice(actions)
        return action, None, None


class RangeAggression(Strategy):
    def __init__(self, *args, **kwargs):
        self.name = "ranged_aggression"

    def sample_action(self, creature, combat_handler):
        """
        Always uses "Arrow Shot" if action available
        :param creature:
        :param combat_handler:
        :return:
        """
        enemy = self.determine_enemy(creature=creature, combat_handler=combat_handler)
        current_state = self.get_raw_state(
            creature=creature,
            enemy=enemy,
            combat_handler=combat_handler
        )
        num_attacks_used = current_state[0][NUM_ATTACKS_USED]
        if num_attacks_used < 1:
            action = creature.get_action("Arrow Shot")
        else:
            action = creature.get_action("end_turn")

        return action, None, None


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
        return action, None, None

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

        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.memory = PrioritizedMemory(memory_length)
        self.name = "DQN"
        self.batch_size = batch_size

        self.learning_rate_decay_freq = TIME_LIMIT * 100
        self.n_learning_rate_decays = 0
        self.n_weight_updates = 0

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

        # Initialize weights
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, self.n_actions)
        )
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    def get_best_action(self, state):
        """
        :param state:
        :return:
        """
        # Note: Perhaps remove .detach()
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
            state = self.get_current_state(creature=creature, combat_handler=combat_handler)

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
        return self.index_to_action[action_index.data.tolist()[0][0]], None, None

    def update_weights(self, predicted_batch, target_batch, emphasis_weights=None):
        self.n_weight_updates += 1
        loss = mean_sq_error(target=target_batch, predicted=predicted_batch, emphasis_weights=emphasis_weights)

        # Zero out accumulated gradients
        self.policy_net.zero_grad()
        # self.optimizer.zero_grad()

        loss.backward()

        # Update weights
        # self.optimizer.step()
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
        if action is None:
            return

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

        return

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

        if next_state is None:
            if not enemy.is_alive():
                reward = 5

        # # Get raw state
        # raw_next_state = self.get_raw_state(creature, enemy, combat_handler)
        #
        # # Damage done
        # damage_done = (current_state - raw_next_state)[0][1]
        # reward += round(float(damage_done), 2) * 10
        #
        # # Damage taken
        # damage_taken = (raw_next_state - current_state)[0][0]
        # reward += round(float(damage_taken), 2) * 10

        return reward


class SARSA(FunctionApproximation):
    """
    SARSA
    """
    def __init__(self, max_training_steps=1e5, epsilon_start=0.9, epsilon_end=0.05, alpha=1e-4,
                 gamma=0.9, update_frequency=5e4, memory_length=16834, batch_size=1024):
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
        batch, indicies, emphasis_weights = self.memory.sample(self.batch_size)
        batch = SARSAExperience(*zip(*batch))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        evaluation_batch = torch.zeros((self.batch_size, 1))

        if non_final_mask.sum() >= 1:
            non_final_next_states = torch.cat(
                filter_out_final_states(batch_data=batch.next_state, non_final_mask=non_final_mask)
            )
            non_final_next_actions = torch.tensor(
                filter_out_final_states(batch_data=batch.next_action, non_final_mask=non_final_mask)
            ).view(-1, 1)
            non_final_evaluation_batch = self.policy_net(non_final_next_states).gather(1, non_final_next_actions)
            evaluation_batch[non_final_mask] = non_final_evaluation_batch

        # Calculate gradients
        target_batch = self.gamma * evaluation_batch + reward_batch
        predicted_batch = self.policy_net(state_batch).gather(1, action_batch)
        self.update_weights(
            predicted_batch=predicted_batch,
            target_batch=target_batch,
            emphasis_weights=emphasis_weights
        )

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
        reward = -0.01

        if next_state is None:
            is_dead = creature.hit_points < 0
            is_too_many_combat_actions = creature.action_count >= TIME_LIMIT
            if not(is_dead or is_too_many_combat_actions):
                reward = 100

        return reward


class DoubleDQN(FunctionApproximation):
    def __init__(self, max_training_steps=1e5, epsilon_start=0.5, epsilon_end=0.05, alpha=1e-2,
                 gamma=0.99, update_frequency=30000, memory_length=16834, batch_size=128):
        super().__init__(
            max_training_steps, epsilon_start, epsilon_end, alpha, gamma, update_frequency, memory_length, batch_size
        )
        self.name = "double_DQN"
        self.optimizer = None
        self.memory = PrioritizedMemory(memory_length, experience_type=Experience)

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)

        h1 = self.n_actions

        # Initialize weights
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, self.n_actions),
        )
        self.target_net = copy.deepcopy(self.policy_net)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    def learn_from_replay(self):
        # Sample experiences from memory
        batch, indicies, emphasis_weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*batch))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        evaluation_batch = torch.zeros((self.batch_size, 1))

        if non_final_mask.sum() >= 1:
            non_final_next_states = torch.cat(
                filter_out_final_states(batch_data=batch.next_state, non_final_mask=non_final_mask)
            )
            selected_actions = self.policy_net(non_final_next_states).max(1)[1].view(-1, 1)
            non_final_evaluation_batch = self.target_net(non_final_next_states).detach().gather(1, selected_actions)
            evaluation_batch[non_final_mask] = non_final_evaluation_batch

        # Calculate gradients
        target_batch = self.gamma * evaluation_batch + reward_batch
        predicted_batch = self.policy_net(state_batch).gather(1, action_batch)
        self.update_weights(
            predicted_batch=predicted_batch,
            target_batch=target_batch,
            emphasis_weights=emphasis_weights
        )

        # Update priorities
        priorities = ((predicted_batch - target_batch) ** 2 + self.memory.epsilon) ** self.memory.alpha
        self.memory.update_priorities(indicies=indicies, priorities=priorities)


class DoubleDuelingDQN(DoubleDQN):
    def __init__(self, max_training_steps=5e6, epsilon_start=0.9, epsilon_end=0.05, alpha=1e-5,
                 gamma=0.99, update_frequency=30000, memory_length=16834, batch_size=128):
        super().__init__(
            max_training_steps, epsilon_start, epsilon_end, alpha, gamma, update_frequency, memory_length, batch_size
        )
        self.name = "double_dueling_DQN"
        self.optimizer = None
        self.memory = PrioritizedMemory(memory_length, experience_type=Experience)
        self.learning_rate_decay_freq = TIME_LIMIT * 50

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)

        h = self.n_actions * 2

        # Initialize weights
        self.policy_net = DuelingNet(
            n_features=self.n_states,
            n_hidden_units=h,
            n_outputs=self.n_actions
        )
        self.target_net = copy.deepcopy(self.policy_net)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)


class MCDoubleDuelingDQN(DoubleDuelingDQN):
    def __init__(self, max_training_steps=1e5, epsilon_start=0.5, epsilon_end=0.05, alpha=1e-2,
                 gamma=0.90, update_frequency=30000, memory_length=16834, batch_size=128):
        super().__init__(
            max_training_steps, epsilon_start, epsilon_end, alpha, gamma, update_frequency, memory_length, batch_size
        )
        self.name = "mc_double_dueling_dqn"
        self.optimizer = None

    def update_step(self, action, creature, current_state, next_state, combat_handler):
        pass

    def calculate_g_t(self, trajectory):
        discounted_rewards = list()

        # Obtain rewards
        rewards = [t[2] for t in trajectory]
        rewards.reverse()

        # Calculate discounted sum of rewards
        for reward in rewards:
            if discounted_rewards:
                discounted_reward = reward + self.gamma * discounted_rewards[-1]
            else:
                discounted_reward = reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards.reverse()

        return discounted_rewards

    def update_step_trajectory(self, trajectory):
        g_t = self.calculate_g_t(trajectory)

        for t, (current_state, action, reward, next_state) in enumerate(trajectory):
            target = torch.tensor(g_t[t]).float()
            current_state = torch.tensor(current_state).float()
            action_index = torch.tensor([[self.action_to_index[action]]])
            predicted = self.policy_net(current_state).gather(1, action_index)
            self.update_weights(
                predicted_batch=predicted,
                target_batch=target,
                emphasis_weights=None
            )


class PPO(FunctionApproximation):
    def __init__(self, max_training_steps=1e5, epsilon_start=0.5, epsilon_end=0.05, alpha=1e-5,
                 gamma=0.99, update_frequency=30000, memory_length=16834, batch_size=128):
        super().__init__(
            max_training_steps, epsilon_start, epsilon_end, alpha, gamma, update_frequency, memory_length, batch_size
        )
        self.name = "PPO"
        self.optimizer = None

    def initialize_weights(self, creature, state):
        self.n_states = state.shape[1]
        self.n_actions = len(creature.actions)

        h = self.n_actions

        # Initialize weights
        self.policy_net = ActorCritic(
            n_features=self.n_states,
            n_hidden_units=h,
            n_outputs=self.n_actions
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    def update_step(self, action, creature, current_state, next_state, combat_handler):
        pass

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
            state = self.get_current_state(creature=creature, combat_handler=combat_handler)

        dist, value = self.policy_net(state)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        action = self.index_to_action[action_index.data.numpy()[0]]

        # Return action
        return action, log_prob, value

    def evaluate_state_and_action(self, creature, combat_handler, state, action):
        """
        Obtain:
           - the probability of selection `action_index` when in input state 'state'
           - the value of the being in input state `state`
        :param creature:
        :param combat_handler:
        :param action:
        :param state:
        :return:
        """
        # Obtain state and action index:
        action_index = self.action_to_index.get(action)

        # Check if creature hadn't taken any actions.
        if action_index is None:
            return

        # Convert to tensor
        action_index = torch.tensor(action_index)

        # Check if end of combat state
        if state is None:
            state = self.get_current_state(creature=creature, combat_handler=combat_handler)

        dist, value = self.policy_net(state)
        log_prob = dist.log_prob(action_index)
        return log_prob, value

    def get_gae(self, trajectory, lmbda=0.95):
        """
        :param trajectory:
        :param lmbda:
        :return:
        """
        # Todo: replace this codeblock
        rewards = [t[2] for t in trajectory]
        values = [t[-1] for t in trajectory]
        dummy_next_value = 0  # should get masked out
        values = values + [dummy_next_value]
        masks = [t[3] is not None for t in trajectory]

        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * lmbda * masks[step] * gae
            returns.insert(0, gae + values[step])

        returns = torch.cat(returns)
        return returns

    def get_returns(self, trajectory):
        rewards = [t[2] for t in trajectory]
        is_terminals = [t[3] is None for t in trajectory]
        discounted_rewards = list()

        for reward, is_terminal in reversed(list(zip(rewards, is_terminals))):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, [discounted_reward])

        discounted_rewards = torch.tensor(discounted_rewards)

        return discounted_rewards

    @staticmethod
    def select_random_batch(current_states, actions, log_probs,returns, advantages, mini_batch_size):
        random_indicies = np.random.randint(0, len(current_states), mini_batch_size)

        batch_current_states = current_states[random_indicies]
        batch_actions = actions[random_indicies]
        batch_log_probs = log_probs[random_indicies]
        batch_returns = returns[random_indicies]
        batch_advantages = advantages[random_indicies]

        return batch_current_states, batch_actions, batch_log_probs, batch_returns, batch_advantages

    def update_step_trajectory(self, trajectory, clip_val=0.2):
        """
        Todo: Make sure trajectory contains 'value's
        :param trajectory:
        :param clip_val:
        :return:
        """
        if trajectory == [None]:
            return
        returns = self.get_gae(trajectory)
        old_values = torch.tensor([[traj[VALUE_INDEX]] for traj in trajectory])
        advantages = returns - old_values

        current_states, actions, rewards, next_states, old_log_probs, values = list(zip(*trajectory))
        current_states = torch.cat(current_states)
        old_log_probs = torch.cat(old_log_probs)
        action_indicies = torch.tensor([[self.action_to_index[action]] for action in actions])

        # Learn for each step in trajectory
        for _ in range(len(trajectory)):
            # Get random sample of experienes
            batch_current_state, batch_action_indicies, batch_old_log_probs, batch_returns, batch_advantages = \
                self.select_random_batch(
                    current_states=current_states,
                    actions=action_indicies,
                    log_probs=old_log_probs,
                    returns=returns,
                    advantages=advantages,
                    mini_batch_size=16
                )
            batch_old_log_probs = batch_old_log_probs.detach()
            batch_current_state = batch_current_state.detach()
            batch_action_indicies = batch_action_indicies.detach()

            new_log_probs, new_values, entropy = self.policy_net.evaluate(
                batch_current_state,
                batch_action_indicies
            )

            # Calculate loss for actor
            ratio = (new_log_probs - batch_old_log_probs.detach()).exp().view(-1, 1)
            loss1 = ratio * batch_advantages.detach()
            loss2 = torch.clamp(ratio, 1 - clip_val, 1 + clip_val) * batch_advantages.detach()
            actor_loss = -torch.min(loss1, loss2).mean()

            # Calculate loss for critic
            sampled_returns = batch_returns.detach()
            critic_loss = (new_values - sampled_returns).pow(2).mean()

            # Credit: https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
            overall_loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            self.optimizer.zero_grad()
            overall_loss.backward()
            self.optimizer.step()

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

        if next_state is None:
            if not enemy.is_alive():
                reward = 5

        # # Get raw state
        # raw_next_state = self.get_raw_state(creature, enemy, combat_handler)

        # # Damage done
        # damage_done = (current_state - raw_next_state)[0][1]
        # reward += round(float(damage_done), 2) * 10

        # # Damage taken
        # damage_taken = (raw_next_state - current_state)[0][0]
        # reward += round(float(damage_taken), 2) * 10

        return reward






