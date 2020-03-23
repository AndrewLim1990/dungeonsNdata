from creatures import leotris
from creatures import vampire
from environments import square_room
from combat_handler import CombatHandler

import curses
import sys
import time
import traceback


def main():
    """
    Todo: Provide main documentation/overview
    """
    # Try to obtain console for visualization
    try:
        console = curses.initscr()
    except Exception as e:
        print(e)
        console = None

    # Attempt
    try:
        combat_handler = CombatHandler(
            environment=square_room,
            combatants=[leotris, vampire],
            console=console
        )
        combat_handler.run()
        time.sleep(5)
    except Exception as e:
        print(e)
        traceback.print_exc(file=sys.stdout)

    if console:
        curses.endwin()


if __name__ == "__main__":
    main()
