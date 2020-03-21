import time

from utils import draw_location
from actions import Attack


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
    def __init__(self, environment, console, combatants=[]):
        self.environment = environment
        self.combatants = combatants
        self.turn_order = list()
        self.combat_is_over = False
        self.console = console
        self.first_visualization = True

    def add_combatant(self, combatant):
        self.combatants.append(combatant)

    def remove_combatant(self, combatant):
        self.combatants.remove(combatant)

    def set_environment(self, environment):
        self.environment = environment

    def initialize_combat(self):
        self.combat_is_over = False
        self.roll_combat_initiative()

    def roll_combat_initiative(self):
        """
        Determines turn order of all combatants
        """
        for combatant in self.combatants:
            self.turn_order.append([combatant, combatant.roll_initiative()])
        self.turn_order = sorted(self.turn_order, key=lambda x: x[1], reverse=True)

        # print("Turn order: {}".format([(c.name, initiative) for c, initiative in self.turn_order]))

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
        num_combatants = len(self.combatants)
        self.combat_is_over = num_combatants <= 1
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

    def visualize(self, creature, old_location):
        """
        Visualizes state of battlefield
        """
        if self.first_visualization:
            self.environment.draw_board(self.console)
            self.first_visualization = False

        # Clear old location
        draw_location(
            self.console,
            x=int(old_location[0] / 5),
            y=int(old_location[1] / 5),
            char=" "
        )

        # Draw new location
        draw_location(
            self.console,
            x=int(creature.location[0] / 5),
            y=int(creature.location[1] / 5),
            char=creature.symbol
        )

        time.sleep(0.25)

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

    def run(self):
        """
        Runs Combat
        """
        self.initialize_combat()
        while not self.check_if_combat_is_over():
            for combatant, initiative in self.turn_order:
                starting_location = combatant.location
                _, meta_data_list = combatant.player.take_action(creature=combatant, combat_handler=self)

                # Visualize movement
                if self.console:
                    self.visualize(creature=combatant, old_location=starting_location)
                    self.report_combat(meta_data_list)

            self.end_of_round_cleanup()
            if self.check_if_combat_is_over():
                break


