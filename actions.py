from utils import roll_dice
from utils import calculate_distance

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
    def __init__(self, hit_bonus, damage_bonus, num_damage_dice, damage_dice, range, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_bonus = hit_bonus
        self.damage_bonus = damage_bonus
        self.num_damage_dice = num_damage_dice
        self.damage_dice = damage_dice
        self.range = range

    def use(self, source_creature, target_creature):
        """
        Uses an attack to hit a target creature
        """
        # Check for illegal attacks
        is_under_attacks_allowed = source_creature.attacks_used < source_creature.attacks_allowed
        distance = calculate_distance(source_creature.location, target_creature.location)
        is_in_range = distance <= self.range
        if not is_in_range:
            print("ILLEGAL ACTION: Target is not within range. \n  source(: {}\n  target: {}\n  distance: {}".format(
                source_creature.location,
                target_creature.location,
                distance
            ))
            return
        elif not is_under_attacks_allowed :
            print("ILLEGAL ACTION: {}/{} attacks already used.".format(source_creature.attacks_used, source_creature.attacks_allowed))
            return

        # Legal attack:
        hit_roll = roll_dice(TWENTY_SIDED_DICE) + self.hit_bonus
        source_creature.attacks_used += 1

        if hit_roll >= target_creature.armor_class:
            damage = np.sum([roll_dice(self.damage_dice) for _ in range(self.num_damage_dice)])
            target_creature.hit_points -= damage
            print("{}: {} hits with {} points of damage. {} is down to {}HP".format(source_creature.name, hit_roll, damage, target_creature.name, target_creature.hit_points))
        else:
            print("{} missed with a hit roll of {}. {} has {}AC.".format(source_creature.name, hit_roll, target_creature.name, target_creature.armor_class))


class Move(Action):
    def __init__(self, direction, distance, environment, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = direction
        self.distance = distance
        self.environment = environment

    def use(self, source_creature):
        """
        Moves the character
        """
        has_movement = source_creature.movement_remaining >= self.distance

        if not has_movement:
            print("ILLEGAL ACTION: Creature has no movement left")

        target_x = source_creature.location[0]
        target_y = source_creature.location[1]

        if self.direction == "right":
            target_x += self.distance
        elif self.direction == "left":
            target_x -= self.distance
        elif self.direction == "up":
            target_y += self.distance
        elif self.direction == "down":
            target_y -= self.distance

        is_legal_target_location = self.environment.check_if_legal(target_location=np.array([target_x, target_y]))
        if is_legal_target_location:
            old_location = source_creature.location
            source_creature.location = np.array([target_x, target_y])
            source_creature.movement_remaining -= self.distance
            print("{} has moved from {} to {}".format(source_creature.name, old_location, source_creature.location))
        else:
            print("Location [{}, {}] is not legal in the environment: {}")


# Todo: Move into DB
vampire_bite = Attack(hit_bonus=10, damage_bonus=10, num_damage_dice=2, damage_dice=6, range=5)


