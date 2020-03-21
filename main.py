from creatures import leotris
from creatures import vampire
from environments import square_room
from combat_handler import CombatHandler

import curses
import sys
import time
import traceback


def main():
    try:
        console = curses.initscr()
    except Exception as e:
        console = None

    try:
        combat_handler = CombatHandler(
            environment=square_room,
            combatants=[leotris, vampire],
            console=console
        )
        combat_handler.run()
        time.sleep(5)
        if console:
            curses.endwin()
    except Exception as e:
        if console:
            curses.endwin()
        print(e)
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
