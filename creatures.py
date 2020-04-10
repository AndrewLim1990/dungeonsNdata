from actions import vampire_bite
from actions import sword_slash
from actions import arrow_shot
from actions import EndTurn
from actions import MoveLeft
from actions import MoveRight
from actions import MoveUp
from actions import MoveDown
from player_strategy import dungeon_master
from player_strategy import hayden
from utils.dnd_utils import roll_dice

import numpy as np


class Creature:
    """
    Represents a creature
    """
    def __init__(
            self, player, name, hit_points, armor_class, location,
            speed=30, actions=[], reactions=[], attacks_allowed=1,
            spells_allowed=1, symbol="x"):
        self.player = player
        self.name = name
        self.hit_points = hit_points
        self.max_hit_points = hit_points
        self.armor_class = armor_class
        self.speed = speed
        self.movement_remaining = self.speed
        self.actions = [MoveLeft(), MoveRight(), MoveUp(), MoveDown(), EndTurn()] + actions
        self.reactions = reactions
        self.location = location
        self.attacks_allowed = attacks_allowed
        self.spells_allowed = spells_allowed
        self.attacks_used = 0
        self.spells_used = 0
        self.actions_used = 0
        self.bonus_actions_used = 0
        self.symbol = symbol

    def use_action(self, action, **kwargs):
        """
        Uses action
        """
        return action.use(self, **kwargs)

    def roll_initiative(self):
        """
        Roll initiative
        Todo: Add modifier
        """
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



# Todo: Move into DB
vampire = Creature(
    player=dungeon_master,
    name="Strahd",
    hit_points=100,
    armor_class=17,
    actions=[vampire_bite],
    location=np.array([5, 5]),
    symbol="@"
)

leotris = Creature(
    player=hayden,
    name="Leotris",
    hit_points=25,
    armor_class=16,
    actions=[arrow_shot, sword_slash],
    location=np.array([5, 10]),
    symbol="x"
)
magnus = Creature(
    player=grant,
    name="Magnus",
    hit_points=21,
    armor_class=18,
    actions=[hammer_smash],
    location=np.array([3, 8]),
    symbol="y"
)