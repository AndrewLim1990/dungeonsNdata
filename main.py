from creatures import leotris
from creatures import vampire
from actions import vampire_bite
from actions import Move
from environments import square_room
from combat_handler import CombatHandler


def main():
    # vampire.use_action(Move(direction="right", distance=15, environment=square_room))
    # vampire.use_action(vampire_bite, target_creature=leotris)
    combat_handler = CombatHandler(environment=square_room, combatants=[leotris, vampire])
    combat_handler.run()


if __name__ == "__main__":
    main()
