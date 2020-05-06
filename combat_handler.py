import numpy as np
import torch

from actions import Attack
from actions import EndTurn
from collections import defaultdict
from collections import Counter
from utils.dnd_utils import draw_location

TIME_LIMIT = 1500


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
    def __init__(self, environment, combatants=[]):
        self.environment = environment
        self.combatants = combatants
        self.turn_order = list()
        self.combat_is_over = False
        self.first_visualization = True
        self.current_turn = None
        self.last_known_states = dict()
        self.total_q_val = defaultdict(list)

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
            enemy = combatant.player.strategy.determine_enemy(combatant, self)
            self.last_known_states[combatant] = self.get_current_state(combatant, enemy)

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
            # exceeded_time_limit = [creature.action_count >= TIME_LIMIT for creature in self.combatants].any()
            exceeded_time_limit = leotris.action_count >= TIME_LIMIT
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

    def get_raw_state(self, creature, enemy):
        current_state = np.array([[
            creature.hit_points / creature.max_hit_points,              # creature health
            enemy.hit_points / enemy.max_hit_points,                    # enemy health
            creature.location[0] / self.environment.room_width,         # creature x loc
            creature.location[1] / self.environment.room_length,        # creature y loc
            enemy.location[0] / self.environment.room_width,            # enemy x loc
            enemy.location[1] / self.environment.room_length,           # enemy y loc
            creature.attacks_used,                                      # can attack
            creature.movement_remaining / creature.speed,               # can move
            (2 * creature.action_count - TIME_LIMIT) / TIME_LIMIT       # num actions taken
        ]])

        return current_state

    def get_current_state(self, creature, enemy):
        creature_hitpoints_idx = 0
        enemy_hitpoints_idx = 1
        current_state = self.get_raw_state(creature, enemy)

        is_end_state = (current_state[0][creature_hitpoints_idx] <= 0) or (current_state[0][enemy_hitpoints_idx] <= 0)
        is_timed_out = current_state[0][-1] >= 1
        if is_end_state or is_timed_out:
            return None

        return current_state

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

        while not self.check_if_combat_is_over():
            for combatant, initiative in self.turn_order:
                # Update combatants if it gets back to their turn after they ended
                if (self.last_known_states.get(combatant) is not None) and not is_first_step:
                    enemy = combatant.player.strategy.determine_enemy(combatant, combat_handler=self)
                    next_state = self.get_current_state(creature=combatant, enemy=enemy)
                    combatant.player.strategy.update_step(
                        action=combatant.get_action("end_turn"),
                        creature=combatant,
                        current_state=self.last_known_states[combatant],
                        next_state=next_state,
                        combat_handler=self
                    )

                while True:
                    # Poll for action to use
                    action, q_val = combatant.player.strategy.sample_action(
                        creature=combatant,
                        combat_handler=self
                    )
                    self.total_q_val[combatant].append(q_val)

                    # Perform action (update state, update combat handler)
                    enemy = combatant.player.strategy.determine_enemy(combatant, combat_handler=self)
                    current_state = self.get_current_state(
                        creature=combatant, enemy=enemy
                    )
                    combatant.use_action(
                        action,
                        combat_handler=self,
                        target_creature=combatant.sample_enemy(combat_handler=self)
                    )
                    next_state = self.get_current_state(
                        creature=combatant, enemy=enemy
                    )
                    self.last_known_states[combatant] = next_state

                    # Allow combatants to change strategy
                    if action != combatant.get_action("end_turn"):
                        combatant.player.strategy.update_step(
                            action=action,
                            creature=combatant,
                            current_state=current_state,
                            next_state=next_state,
                            combat_handler=self
                        )
                    if self.check_if_combat_is_over():
                        # Let loser adjust
                        loser = [creature for creature in self.combatants if creature.hit_points <= 0]
                        if loser:
                            loser = loser[0]
                            winner = [creature for creature in self.combatants if creature.hit_points > 0][0]
                            current_state = self.get_current_state(creature=loser, enemy=winner)

                            loser.player.strategy.update_step(
                                loser.get_action("end_turn"),
                                creature=loser,
                                current_state=self.last_known_states[loser],
                                next_state=current_state,
                                combat_handler=self
                            )

                        end_now = True
                        break

                    # End of combatant turn:
                    if type(action) == EndTurn:
                        break

                # Combat has ended:
                if end_now:
                    lst_state = self.get_raw_state(
                        creature=self.get_combatant("Leotris"),
                        enemy=self.get_combatant("Strahd")
                    )[0]
                    leotris = self.get_combatant("Leotris")
                    end_now = False
                    break

            self.end_of_round_cleanup()
            is_first_step = False

        avg_q_val = torch.tensor(self.total_q_val[leotris]).mean()

        remaining_combatants = [creature.name for creature in self.combatants]
        winner = remaining_combatants[0]
        if len(remaining_combatants) > 1:
            winner = "Timeout"
        # print("Winner: {}".format(winner))
        return winner, avg_q_val, lst_state, leotris.action_count
