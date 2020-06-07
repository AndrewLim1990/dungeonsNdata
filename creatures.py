from agents import DoubleDQN
from agents import DoubleDuelingDQN
from agents import MCDoubleDuelingDQN
from agents import PPO
from agents import RangeAggression
from agents import RandomStrategy
from agents import SARSA
from actions import vampire_bite
from actions import arrow_shot
from actions import EndTurn
from actions import MoveLeft
from actions import MoveRight
from actions import MoveUp
from actions import MoveDown
from players import dungeon_master
from players import hayden
from utils.dnd_utils import roll_dice

import numpy as np


class Creature:
    """
    Represents a creature
    """
    def __init__(
            self, player, name, hit_points, armor_class, location, strategy,
            speed=30, actions=[], reactions=[], attacks_allowed=1,
            spells_allowed=1, symbol="x"):
        self.player = player
        self.name = name
        self.hit_points = hit_points
        self.max_hit_points = hit_points
        self.armor_class = armor_class
        self.speed = speed
        self.movement_remaining = self.speed
        self.strategy = strategy
        self.actions = [EndTurn()] + actions
        self.reactions = reactions
        self.location = location
        self.attacks_allowed = attacks_allowed
        self.spells_allowed = spells_allowed
        self.attacks_used = 0
        self.spells_used = 0
        self.actions_used = 0
        self.bonus_actions_used = 0
        self.symbol = symbol
        self.action_count = 0

    def use_action(self, action, **kwargs):
        """
        Uses action
        """
        self.action_count += 1
        combat_handler = kwargs['combat_handler']
        combat_handler.actions_this_round[self] += 1
        return action.use(self, **kwargs)

    def roll_initiative(self):
        """
        Roll initiative
        Todo: Add modifier
        """
        self.action_count = 0
        modifier = 0
        return roll_dice(20) + modifier

    def reset_round_resources(self):
        """
        Resets the following:
            * Movement
            * Number of used attacks
            * Number of used spells
            * Number of used actions
            * Number of used bonus actions
        """

        self.attacks_used = 0
        self.spells_used = 0
        self.actions_used = 0
        self.bonus_actions_used = 0
        self.movement_remaining = self.speed

    def is_alive(self):
        """
        :return:
        """
        if self.hit_points > 0:
            is_alive = True
        else:
            is_alive = False
        return is_alive

    def sample_enemy(self, combat_handler):
        """
        todo: remove self from potential enemies
        :param combat_handler:
        :return:
        """
        creatures = combat_handler.combatants
        creatures = [creature for creature in creatures if creature.name != self.name]
        random_enemy = np.random.choice(creatures)
        return random_enemy

    def full_heal(self):
        self.hit_points = self.max_hit_points

    def get_action(self, name):
        """
        :param name:
        :return:
        """
        matching_actions = [action for action in self.actions if action.name == name]
        assert len(matching_actions) == 1, "Exactly 1 action must match the given action name"
        matching_action = matching_actions[0]
        return matching_action

    def initialize(self, combat_handler):
        self.strategy.initialize(creature=self, combat_handler=combat_handler)


# Todo: Move into DB
vampire = Creature(
    player=dungeon_master,
    name="Strahd",
    hit_points=200,
    armor_class=17,
    actions=[MoveLeft(), MoveRight(), MoveUp(), MoveDown(), vampire_bite],
    location=np.array([5, 5]),
    symbol="@",
    strategy=RandomStrategy()
)

leotris = Creature(
    player=hayden,
    name="Leotris",
    hit_points=25,
    armor_class=16,
    actions=[MoveLeft(), MoveRight(), MoveUp(), MoveDown(), arrow_shot],
    location=np.array([5, 10]),
    symbol="x",
    strategy=PPO()
)