from actions import EndTurn
from collections import defaultdict

import numpy as np


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
        self.combat_is_over = False
        self.first_visualization = True
        self.current_turn = None
        self.last_known_states = dict()
        self.total_q_val = defaultdict(list)
        self.time_limit = time_limit

    def add_combatant(self, combatant):
        self.combatants.append(combatant)

    def remove_combatant(self, combatant):
        self.combatants.remove(combatant)

    def set_environment(self, environment):
        self.environment = environment

    def initialize_combat(self):
        self.combat_is_over = False
        self.roll_combat_initiative()
        self.current_turn = self.turn_order[0][0]
        self.full_heal_combatants()
        for combatant in self.combatants:
            self.last_known_states[combatant] = combatant.player.strategy.get_current_state(
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
        self.remove_dead_combatants()

    def check_if_combat_is_over(self):
        exceeded_time_limit = False
        num_live_combatants = len([creature for creature in self.combatants if creature.hit_points > 0])
        try:
            leotris = self.get_combatant("Leotris")
            exceeded_time_limit = leotris.action_count >= self.time_limit
        except IndexError:
            pass
        self.combat_is_over = (num_live_combatants <= 1) or exceeded_time_limit
        if self.combat_is_over:
            return True
        else:
            return False

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

    def run(self):
        """
        Runs Combat
        """
        self.initialize_combat()
        end_now = False
        is_first_step = True
        accumulated_reward = defaultdict(list)

        while not self.check_if_combat_is_over():
            for combatant, initiative in self.turn_order:
                # Update combatants if it gets back to their turn after they ended
                if (self.last_known_states.get(combatant) is not None) and not is_first_step:
                    next_state = combatant.player.strategy.get_current_state(creature=combatant, combat_handler=self)
                    reward = combatant.player.strategy.update_step(
                        action=combatant.get_action("end_turn"),
                        creature=combatant,
                        current_state=self.last_known_states[combatant],
                        next_state=next_state,
                        combat_handler=self
                    )
                    accumulated_reward[combatant.name].append(reward)

                while True:
                    # Poll for action to use
                    action, q_val = combatant.player.strategy.sample_action(
                        creature=combatant,
                        combat_handler=self
                    )
                    self.total_q_val[combatant].append(q_val)

                    # Perform action (update state, update combat handler)
                    current_state = combatant.player.strategy.get_current_state(creature=combatant, combat_handler=self)
                    combatant.use_action(
                        action,
                        combat_handler=self,
                        target_creature=combatant.sample_enemy(combat_handler=self)
                    )
                    next_state = combatant.player.strategy.get_current_state(creature=combatant, combat_handler=self)
                    self.last_known_states[combatant] = next_state

                    # Allow combatants to change strategy
                    if action != combatant.get_action("end_turn"):
                        reward = combatant.player.strategy.update_step(
                            action=action,
                            creature=combatant,
                            current_state=current_state,
                            next_state=next_state,
                            combat_handler=self
                        )
                        accumulated_reward[combatant.name].append(reward)

                    if self.check_if_combat_is_over():
                        # Let loser adjust
                        loser = [creature for creature in self.combatants if creature.hit_points <= 0]
                        if loser:
                            loser = loser[0]
                            current_state = combatant.player.strategy.get_current_state(
                                creature=combatant, combat_handler=self
                            )

                            reward = loser.player.strategy.update_step(
                                loser.get_action("end_turn"),
                                creature=loser,
                                current_state=self.last_known_states[loser],
                                next_state=current_state,
                                combat_handler=self
                            )
                            accumulated_reward[loser.name].append(reward)

                        end_now = True
                        break

                    # End of combatant turn:
                    if type(action) == EndTurn:
                        break

                # Combat has ended:
                if end_now:
                    lst_state = self.get_combatant("Leotris").player.strategy.get_raw_state(
                        creature=self.get_combatant("Leotris"),
                        enemy=self.get_combatant("Strahd"),
                        combat_handler=self
                    )[0]
                    leotris = self.get_combatant("Leotris")
                    end_now = False
                    break

            self.end_of_round_cleanup()
            is_first_step = False

        remaining_combatants = [creature.name for creature in self.combatants]
        winner = remaining_combatants[0]
        if len(remaining_combatants) > 1:
            winner = "Timeout"

        avg_reward = np.sum(accumulated_reward.get("Leotris", [0]))
        return winner, avg_reward, lst_state, leotris.action_count
