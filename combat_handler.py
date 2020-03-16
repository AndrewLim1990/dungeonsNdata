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
    def __init__(self, environment, combatants=[]):
        self.environment = environment
        self.combatants = combatants
        self.turn_order = list()
        self.combat_is_over = False

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

        print("Turn order: {}".format([(c.name, initiative) for c, initiative in self.turn_order]))

    def reset_resources(self):
        pass

    def check_if_combat_is_over(self):
        pass

    def run(self):
        """
        Runs Combat
        """
        self.initialize_combat()
        while not self.combat_is_over:
            for combatant, initiative in self.turn_order:
                print("Turn: {}".format(combatant.name))
                combatant_array = np.array(self.combatants)
                is_not_self = combatant_array != combatant
                target_creature = combatant_array[is_not_self][0]
                combatant.use_action(
                    np.random.choice(combatant.actions),
                    environment=self.environment,
                    target_creature=target_creature
                )
            self.combat_is_over = True
