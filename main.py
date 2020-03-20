from creatures import leotris
from creatures import vampire
from environments import square_room
from combat_handler import CombatHandler


def main():
    combat_handler = CombatHandler(
        environment=square_room,
        combatants=[leotris, vampire]
    )
    combat_handler.run()


if __name__ == "__main__":
    # To do:
    #  Make is so people will move different amounts on their turn. Not just 5 feet
    main()
