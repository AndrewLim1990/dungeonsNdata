from settings import END_TURN_SIGNAL

import numpy as np


class Strategy:
    pass


class Player:
    def __init__(self, name, strategy, creatures=[]):
        self.name = name
        self.strategy = strategy
        self.creatures = creatures

    def get_creatures(self, combat_handler):
        """
        Returns a list of creatures belonging to the player
        """
        creatures = list()
        for creature in combat_handler.combatants:
            if creature.owner == self:
                creatures.append(creature)
        return creatures

    def update_creatures(self, combat_handler):
        """
        Updates creature list
        """
        self.creatures = self.get_creatures()

    def add_creature(self, creature):
        self.creatures.append(creature)

    def take_action(self, creature, combat_handler):
        """
        Todo: Fill in method documentation
        """
        self.strategy.take_action(creature, combat_handler)


class PlayerCharacter(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def take_action(self, creature, combat_handler):
        """
        Takes random action for the given creature

        Args:
            creature: Creature who's turn it is
            combat_handler: Contains environment, turn order, combatants, etc
        """
        player = creature.player
        print("Turn: {}({})".format(creature.name, player.name))
        creature_array = np.array(combat_handler.combatants)
        is_not_self = creature_array != creature
        target_creature = creature_array[is_not_self][0]

        while True:
            action_signal = creature.use_action(
                np.random.choice(creature.actions),
                environment=combat_handler.environment,
                target_creature=target_creature
            )

            if action_signal == END_TURN_SIGNAL:
                print("ENDING ENDING ENDING ENDING ENDING ENDING ENDING ENDING ENDING ENDING ENDING ENDING ENDING")
                return


hayden = PlayerCharacter(strategy=RandomStrategy(), name="Hayden")
dungeon_master = PlayerCharacter(strategy=RandomStrategy(), name="Andrew")
