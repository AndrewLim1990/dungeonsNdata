from actions import vampire_bite


class Creature:
    """
    Represents a creature
    """
    def __init__(self, name, hit_points, armor_class, speed=30, actions=[], reactions=[]):
        self.name = name
        self.hit_points = hit_points
        self.armor_class = armor_class
        self.speed = speed
        self.actions = actions
        self.reactions = reactions

    def use_action(self, action, **kwargs):
        """
        Uses action
        """
        action.use(**kwargs)


# Todo: Move into DB
vampire = Creature(
    name="Strahd",
    hit_points=100,
    armor_class=19,
    actions=[vampire_bite]
)

leotris = Creature(
    name="Leotris",
    hit_points=25,
    armor_class=16,
    actions=[vampire_bite]
)