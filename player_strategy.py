from utils.agent_utils import filter_illegal_actions
from agents import QLearningTabularAgent
from agents import DoubleDQN
from agents import DQN
import numpy as np


class Strategy:
    def determine_enemy(self, creature, combat_handler):
        enemy = None
        combatants = combat_handler.combatants
        for combatant in combatants:
            if combatant != creature:
                enemy = combatant
        return enemy

    def update_step(self, *args, **kwargs):
        pass

    def update_strategy(self):
        pass

    def initialize(self, * args, **kwargs):
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
        self.name = "random"

    def sample_action(self, creature, combat_handler):
        actions = filter_illegal_actions(creature=creature, actions=creature.actions)
        action = np.random.choice(actions)
        return action


class RangeAggression(Strategy):
    def __init__(self, *args, **kwargs):
        self.name = "ranged_aggression"

    def sample_action(self, creature, combat_handler):
        """
        Always uses "Arrow Shot"
        :param creature:
        :param combat_handler:
        :return:
        """
        actions = [creature.get_action("Arrow Shot"), creature.get_action("end_turn")]

        action = np.random.choice(actions, p=[0.95, 0.05])

        return action

hayden = PlayerCharacter(strategy=DoubleDQN(), name="Hayden")
# hayden = PlayerCharacter(strategy=DQN(), name="Hayden")
# hayden = PlayerCharacter(strategy=RandomStrategy(), name="Hayden")
# hayden = PlayerCharacter(strategy=RangeAggression(), name="Hayden")
dungeon_master = PlayerCharacter(strategy=RandomStrategy(), name="Andrew")
