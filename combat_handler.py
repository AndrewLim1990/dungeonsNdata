import numpy as np
import time

from actions import Attack
from actions import EndTurn
from utils.dnd_utils import draw_location


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
        num_live_combatants = len([creature for creature in self.combatants if creature.hit_points > 0 ])
        self.combat_is_over = num_live_combatants <= 1
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

    def report_combat(self, meta_data_list):
        """
        Reports any damage
        """
        damage_reports = []
        for meta_data in meta_data_list:
            if meta_data:
                if meta_data.get("type") == Attack:
                    damage_reports.append(meta_data)

        final_damage_report = ""
        for damage_report in damage_reports:
            source_creature = damage_report["source_creature"]
            target_creature = damage_report["target_creature"]
            damage = damage_report["damage"]
            hit_roll = damage_report["hit_roll"]

            if hit_roll >= target_creature.armor_class:
                final_damage_report += "{} rolled a {}/{}AC. {} damage done to {}. {}/{} HP left.\n".format(
                    source_creature.name,
                    hit_roll,
                    target_creature.armor_class,
                    damage,
                    target_creature.name,
                    target_creature.hit_points,
                    target_creature.max_hit_points
                )
            if target_creature.hit_points <= 0:
                final_damage_report += "{} has killed {}.".format(source_creature.name, target_creature.name)

        draw_location(
            self.console,
            x=0,
            y=int(self.environment.room_length / 5) + 2,
            char=str(final_damage_report)
        )

    def let_combatants_update(self, combatant, input_action, current_state, next_state):
        """
        :return:
        """
        players = [combatant.player for combatant in self.combatants]
        players = list(set(players))
        for player in players:
            for creature in player.get_creatures(combat_handler=self):
                # If not your turn, you can still learn:
                if combatant != creature:
                    action = creature.get_action("end_turn")
                else:
                    action = input_action

                player.strategy.update_step(
                    action,
                    creature=creature,
                    current_state=current_state,
                    next_state=next_state,
                    combat_handler=self
                )

    def get_current_state(self, creature, enemy):
        current_state = (
            creature.location[0],  # creature x loc
            creature.location[1],  # creature y loc
            creature.attacks_used < creature.attacks_allowed,  # can attack
            creature.movement_remaining > 0,  # can move
            # creature.is_alive(),  # creature is alive?
            enemy.location[0],  # enemy x loc
            enemy.location[1],  # enemy y loc
            enemy.hit_points,  # enemy health
            # enemy.is_alive()  # enemy is alive?
        )
        return current_state

    def full_heal_combatants(self):
        """
        Fully heals all combatants
        :return: None
        """
        [combatant.full_heal() for combatant in self.combatants]

    def run(self):
        """
        Runs Combat
        """
        self.initialize_combat()
        end_now = False
        # print("---------------------------- NEW ROUND ----------------------------")
        while not self.check_if_combat_is_over():
            for combatant, initiative in self.turn_order:
                # print("--------------------> Turn: {}".format(combatant.name))
                while True:
                    # Poll for action to use
                    action = combatant.player.strategy.sample_action(
                        creature=combatant,
                        combat_handler=self
                    )

                    # Perform action (update state, update combat handler)
                    enemy = combatant.player.strategy.determine_enemy(combatant, combat_handler=self)
                    current_state = self.get_current_state(creature=combatant, enemy=enemy)
                    combatant.use_action(
                        action,
                        combat_handler=self,
                        target_creature=combatant.sample_enemy(combat_handler=self)
                    )
                    next_state = self.get_current_state(creature=combatant, enemy=enemy)

                    # Allow combatants to change strategy
                    # self.let_combatants_update(combatant, action, current_state=current_state, next_state=next_state)
                    combatant.player.strategy.update_step(
                        action,
                        creature=combatant,
                        current_state=current_state,
                        next_state=next_state,
                        combat_handler=self
                    )
                    if self.check_if_combat_is_over():
                        # Let loser adjust
                        loser = [creature for creature in self.combatants if creature.hit_points <= 0][0]
                        winner = [creature for creature in self.combatants if creature.hit_points > 0][0]
                        current_state = self.get_current_state(creature=loser, enemy=winner)
                        if winner.name == "Leotris":
                            pass
                        # print(winner.name)
                        loser.player.strategy.update_step(
                            loser.get_action("end_turn"),
                            creature=loser,
                            current_state=current_state,
                            next_state=next_state,
                            combat_handler=self
                        )
                        end_now = True
                        break
                    if type(action) == EndTurn:
                        break
                if end_now:
                    end_now = False
                    break

            self.end_of_round_cleanup()

        remaining_combatants = [creature.name for creature in self.combatants]
        assert len(remaining_combatants) == 1

        winner = remaining_combatants[0]
        return winner
