from utils import roll_dice
import numpy as np

TWENTY_SIDED_DICE = 20


class Action:
    """
    Represents an action
    """
    pass


class Reaction:
    """
    Represents a reaction
    """
    pass


class AttackOfOpportunity:
    """
    Melee attack as target creature moves away
    """
    pass


class Attack(Action):
    """
    Represents a melee or ranged attack
    """
    def __init__(self, hit_bonus, damage_bonus, num_damage_dice, damage_dice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_bonus = hit_bonus
        self.damage_bonus = damage_bonus
        self.num_damage_dice = num_damage_dice
        self.damage_dice = damage_dice

    def use(self, target_creature):
        """
        Uses an attack to hit a target creature
        """
        hit_roll = roll_dice(TWENTY_SIDED_DICE) + self.hit_bonus

        if hit_roll >= target_creature.armor_class:
            damage = np.sum([roll_dice(self.damage_dice) for _ in range(self.num_damage_dice)])
            target_creature.hit_points -= damage
            print("{} hits with {} points of damage. {} is down to {}HP".format(hit_roll, damage, target_creature.name, target_creature.hit_points))
        else:
            print("Missed with a hit roll of {}. {} has {}AC.".format(hit_roll, target_creature.name, target_creature.armor_class))


# Todo: Move into DB
vampire_bite = Attack(hit_bonus=10, damage_bonus=10, num_damage_dice=2, damage_dice=6)