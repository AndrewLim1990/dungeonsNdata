from creatures import leotris
from creatures import vampire
from environments import square_room
from combat_handler import CombatHandler

import curses


def main():
    console = curses.initscr()
    combat_handler = CombatHandler(
        environment=square_room,
        combatants=[leotris, vampire],
        console=console
    )
    combat_handler.run()
    curses.endwin()


if __name__ == "__main__":
    # To do:
    #  Make is so people will move different amounts on their turn. Not just 5 feet
    main()
