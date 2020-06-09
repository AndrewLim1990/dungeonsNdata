from collections import Counter
from collections import defaultdict
import torch

REWARD_INDEX = 2


class CombatHandler:
    """
    This class implements a combat handler in charge of:
        * Rolling Initiative
        * Turn order
        * Resource resetting at beginning of round
            * Movement
            * Action
            * Bonus Action
    """
    def __init__(self, environment, time_limit, combatants=[]):
        self.environment = environment
        self.combatants = combatants
        self.turn_order = list()
        self.first_visualization = True
        self.last_known_current_states = dict()
        self.last_known_next_states = dict()
        self.last_known_actions = dict()
        self.total_q_val = defaultdict(list)
        self.time_limit = time_limit
        self.actions_this_round = Counter()

    def add_combatant(self, combatant):
        self.combatants.append(combatant)

    def remove_combatant(self, combatant):
        self.combatants.remove(combatant)

    def set_environment(self, environment):
        self.environment = environment

    def initialize_combat(self):
        self.roll_combat_initiative()
        self.full_heal_combatants()
        for combatant in self.combatants:
            self.last_known_current_states[combatant] = combatant.strategy.get_current_state(
                creature=combatant, combat_handler=self
            )
            self.last_known_next_states[combatant] = combatant.strategy.get_current_state(
                creature=combatant, combat_handler=self
            )

    def roll_combat_initiative(self):
        """
        Determines turn order of all combatants
        """
        for combatant in self.combatants:
            self.turn_order.append([combatant, combatant.roll_initiative()])
        self.turn_order = sorted(self.turn_order, key=lambda x: x[1], reverse=True)

    def reset_combat_round_resources(self):
        for combatant in self.combatants:
            combatant.reset_round_resources()

    def remove_dead_combatants(self):
        """
        Todo: Remove if dead - not when "downed" / 0HP
        """
        for combatant in self.combatants:
            if combatant.hit_points <= 0:
                self.remove_combatant(combatant)

    def end_of_round_cleanup(self):
        self.reset_combat_round_resources()

    def combat_is_over(self):
        num_live_combatants = len([creature for creature in self.combatants if creature.hit_points > 0])

        leotris = self.get_combatant("Leotris")
        exceeded_time_limit = leotris.action_count >= self.time_limit

        is_over = (num_live_combatants <= 1) or exceeded_time_limit

        return is_over

    def check_legal_movement(self, target_location):
        """
        Check for clashing into other creatures
        """
        is_legal = True
        for creature in self.combatants:
            if (target_location == creature.location).all():
                is_legal = False
        return is_legal

    def full_heal_combatants(self):
        """
        Fully heals all combatants
        :return: None
        """
        [combatant.full_heal() for combatant in self.combatants]

    def get_combatant(self, name):
        combatant = [c for c in self.combatants if c.name == name][0]
        return combatant

    def determine_winner(self):
        creatures_alive = [creature for creature in self.combatants if creature.hit_points > 0]
        if len(creatures_alive) > 1:
            winner = "Timeout"
        else:
            winner = creatures_alive[0].name

        return winner

    def add_end_turn_sars(self, creature, sars_dict):
        """
        :param creature:
        :return:
        """
        next_state = creature.strategy.get_current_state(creature=creature, combat_handler=self)
        current_state = self.last_known_next_states[creature]

        reward = creature.strategy.determine_reward(
            creature=creature,
            current_state=current_state,
            next_state=next_state,
            combat_handler=self
        )
        action = creature.get_action("end_turn")

        # Evaluate state value and probability of action given state
        log_prob, value = creature.strategy.evaluate_state_and_action(
            creature=creature,
            combat_handler=self,
            state=current_state,
            action=action,
        )

        sars = (current_state, action, reward, next_state, log_prob, value)
        sars_dict[creature].append(sars)

        return sars_dict

    def add_end_combat_sars(self, sars_dict):
        """
        :param sars_dict:
        :return:
        """
        next_state = None
        log_prob = None
        value = None

        for creature in self.combatants:
            current_state = self.last_known_next_states[creature]
            if current_state is None:
                current_state = self.last_known_current_states[creature]
            action = self.last_known_actions.get(creature)
            reward = creature.strategy.determine_reward(
                creature=creature,
                current_state=current_state,
                next_state=next_state,
                combat_handler=self
            )

            # Evaluate state value and probability of action given state
            evaluation = creature.strategy.evaluate_state_and_action(
                creature=creature,
                combat_handler=self,
                state=current_state,
                action=action
            )
            if evaluation is None:
                sars = None
            else:
                log_prob, value = evaluation
                sars = (current_state, action, reward, next_state, log_prob, value)

            sars_dict[creature].append(sars)

        return sars_dict

    def perform_round_step(self, creature):
        current_state = creature.strategy.get_current_state(creature=creature, combat_handler=self)
        action, log_prob, value = creature.strategy.sample_action(creature=creature, combat_handler=self)
        enemy = creature.sample_enemy(combat_handler=self)

        # Keep track of current_state:
        self.last_known_current_states[creature] = current_state

        # Use the action:
        creature.use_action(action=action, target_creature=enemy, combat_handler=self)

        next_state = creature.strategy.get_current_state(creature=creature, combat_handler=self)
        reward = creature.strategy.determine_reward(
            creature=creature,
            current_state=current_state,
            next_state=next_state,
            combat_handler=self
        )

        # Keep track of last known state
        self.last_known_next_states[creature] = next_state
        self.last_known_actions[creature] = action

        return current_state, action, reward, next_state, log_prob, value

    def execute_round(self, round_number):
        """
        Todo: Use action should not need extra kwargs (place in action themselves)
        Todo: use_action should return reward
        :return:
        """
        is_first_round = round_number <= 0
        sars_dict = defaultdict(list)
        self.actions_this_round = Counter()

        for creature, rolled_initiative in self.turn_order:
            # Add end_turn action to sars lists
            if not is_first_round:
                sars_dict = self.add_end_turn_sars(creature=creature, sars_dict=sars_dict)

            while True:
                # Poll and perform action:
                current_state, action, reward, next_state, log_prob, value = self.perform_round_step(creature)

                # If over, exit
                if self.combat_is_over():
                    sars_dict = self.add_end_combat_sars(sars_dict=sars_dict)
                    return sars_dict, self.combat_is_over()

                # If ended turn, go to next creature
                if action.name == "end_turn":
                    break

                # Add results:
                sars = (current_state, action, reward, next_state, log_prob, value)
                sars_dict[creature].append(sars)

        return sars_dict, self.combat_is_over()

    def update_strategies(self, sars_dict):
        """
        :return:
        """
        for creature, sars_list in sars_dict.items():
            if sars_list != [None]:
                for current_state, action, reward, next_state, log_prob, value in sars_list:
                    creature.strategy.update_step(
                        action=action,
                        creature=creature,
                        current_state=current_state,
                        next_state=next_state,
                        combat_handler=self
                    )

    def obtain_info_for_printing(self, creature, total_reward, sars_list=None):
        leotris = self.get_combatant("Leotris")
        strahd = self.get_combatant("Strahd")

        if (sars_list is not None) and (creature == leotris):
            for sars in sars_list:
                reward = 0
                if sars is not None:
                    reward = sars[REWARD_INDEX]
                total_reward += reward

        last_state = leotris.strategy.get_raw_state(creature=leotris, enemy=strahd, combat_handler=self)
        num_actions_used = leotris.action_count

        return total_reward, last_state, num_actions_used

    def run(self):
        """
        Runs combat by:
         - prompting each creature for actions until the "EndTurn" action is selected
         - prompting each creature to learn from the results of the most recent round
         - prompting each creature to learn from the results of the entire trajectory
        """
        self.initialize_combat()
        combat_is_over = False
        round_number = 0
        trajectory_dict = defaultdict(list)
        total_reward = 0

        while not combat_is_over:
            # Run one round of combat (one turn per creature)
            sars_dict, combat_is_over = self.execute_round(round_number)
            for creature, sars_list in sars_dict.items():
                total_reward, last_state, num_actions_used = self.obtain_info_for_printing(
                    creature=creature,
                    total_reward=total_reward,
                    sars_list=sars_list
                )
                trajectory_dict[creature] += sars_list

            # Let creatures update their strategies
            self.update_strategies(sars_dict)

            # Resets round resources (actions/movement used etc)
            self.end_of_round_cleanup()

            round_number += 1

        # Monte carlo updates:
        for creature in self.combatants:
            creature.strategy.update_step_trajectory(trajectory_dict[creature])

        # For reporting
        winner = self.determine_winner()

        return winner, total_reward, last_state, num_actions_used
