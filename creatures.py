from actions import vampire_bite
from actions import sword_slash
from actions import arrow_shot
from actions import MoveLeft
from actions import MoveRight
from actions import MoveUp
from actions import MoveDown
from player_strategy import dungeon_master
from player_strategy import hayden
from utils import roll_dice

import numpy as np

END_TURN_SIGNAL = 42


class Creature:
    """
    Represents a creature
    """
    def __init__(
            self, player, name, hit_points, armor_class, location,
            speed=30, actions=[], reactions=[], attacks_allowed=1,
            spells_allowed=1):
        self.player = player
        self.name = name
        self.hit_points = hit_points
        self.armor_class = armor_class
        self.speed = speed
        self.movement_remaining = self.speed
        self.actions = [MoveLeft(), MoveRight(), MoveUp(), MoveDown()] + actions
        self.reactions = reactions
        self.location = location
        self.attacks_allowed = attacks_allowed
        self.spells_allowed = spells_allowed
        self.attacks_used = 0
        self.spells_used = 0
        self.actions_used = 0
        self.bonus_actions_used = 0

    def use_action(self, action, **kwargs):
        """
        Uses action
        """
        action.use(self, **kwargs)

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

    def take_turn(self, combat_handler, strategy):
        """
        Performs actions during

        Args:
            combat_handler: contains relevant information such as turn order, terrain, locations, and combatants
            strategy: contains strategy to perform
        """
        while True:
            action = strategy.take_action(combatant=self, combat_handler=combat_handler)
            if action == END_TURN_SIGNAL:
                return



# Todo: Move into DB
vampire = Creature(
    player=dungeon_master,
    name="Strahd",
    hit_points=100,
    armor_class=17,
    actions=[vampire_bite],
    location=np.array([0, 0])
)

leotris = Creature(
    player=hayden,
    name="Leotris",
    hit_points=25,
    armor_class=16,
    actions=[arrow_shot],
    location=np.array([25, 25])
)