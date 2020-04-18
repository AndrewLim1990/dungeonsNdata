from utils.agent_utils import filter_illegal_actions

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


class PlayerCharacter(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_action(self, creature, combat_handler):
        actions = filter_illegal_actions(creature=creature, actions=creature.actions)
        action = np.random.choice(actions)
        return action

    def determine_enemy(self, creature, combat_handler):
        enemy = None
        combatants = combat_handler.combatants
        for combatant in combatants:
            if combatant != creature:
                enemy = combatant
        return enemy

    def update_step(self, *args, **kwargs):
        pass

    def initialize(self, * args, **kwargs):
        pass


# hayden = PlayerCharacter(strategy=QLearningTabularAgent(), name="Hayden")
hayden = PlayerCharacter(strategy=RandomStrategy(), name="Hayden")
dungeon_master = PlayerCharacter(strategy=RandomStrategy(), name="Andrew")
