from agents import QLearningTabularAgent
from settings import END_TURN_SIGNAL
from settings import SUCCESSFUL_PLAYER_TURN_SIGNAL

import numpy as np


class Strategy:
    def update_strategy(self):
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
            if creature.player == self:
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
        return self.strategy.take_action(creature, combat_handler)


class PlayerCharacter(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_action(self, creature, combat_handler):
        action = np.random.choice(creature.actions)
        return action

    def determine_enemy(self, creature, combat_handler):
        enemy = None
        combatants = combat_handler.combatants
        for combatant in combatants:
            if combatant != creature:
                enemy = combatant
        return enemy

    def take_action(self, creature, combat_handler):
        """
        Takes random action for the given creature

        Args:
            creature: Creature who's turn it is
            combat_handler: Contains environment, turn order, combatants, etc
        """
        creature_array = np.array(combat_handler.combatants)
        is_not_self = creature_array != creature
        target_creature = creature_array[is_not_self][0]
        meta_data_list = list()

        while True:
            starting_location = creature.location
            action_signal, meta_data = creature.use_action(
                np.random.choice(creature.actions),
                combat_handler=combat_handler,
                target_creature=target_creature
            )
            meta_data_list.append(meta_data)

            if action_signal == END_TURN_SIGNAL:
                meta_data = {"starting_location": starting_location}
                meta_data_list.append(meta_data)
                return SUCCESSFUL_PLAYER_TURN_SIGNAL, meta_data_list

    def update_step(self, *args, **kwargs):
        pass

    def initialize_q(self, *args, **kwargs):
        pass


hayden = PlayerCharacter(strategy=QLearningTabularAgent(), name="Hayden")
# hayden = PlayerCharacter(strategy=RandomStrategy(), name="Hayden")
dungeon_master = PlayerCharacter(strategy=RandomStrategy(), name="Andrew")
