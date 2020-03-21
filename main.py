from creatures import leotris
from creatures import vampire
from environments import square_room
from combat_handler import CombatHandler

import curses
import sys
import traceback



def main():
    console = curses.initscr()
    try:
        combat_handler = CombatHandler(
            environment=square_room,
            combatants=[leotris, vampire],
            console=console
        )
        combat_handler.run()
        curses.endwin()
    except Exception as e:
        curses.endwin()
        print(e)
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    # To do:
    #  Make is so people will move different amounts on their turn. Not just 5 feet
    main()
