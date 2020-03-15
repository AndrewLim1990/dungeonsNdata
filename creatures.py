from actions import vampire_bite
from utils import roll_dice

import numpy as np

class Creature:
    """
    Represents a creature
    """
    def __init__(
            self, name, hit_points, armor_class, location,
            speed=30, actions=[], reactions=[], attacks_allowed=1,
            spells_allowed=1):
        self.name = name
        self.hit_points = hit_points
        self.armor_class = armor_class
        self.speed = speed
        self.movement_remaining = self.speed
        self.actions = actions
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


# Todo: Move into DB
vampire = Creature(
    name="Strahd",
    hit_points=100,
    armor_class=19,
    actions=[vampire_bite],
    location=np.array([0, 0])
)

leotris = Creature(
    name="Leotris",
    hit_points=25,
    armor_class=16,
    actions=[vampire_bite],
    location=np.array([0, 5])
)