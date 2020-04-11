from creatures import leotris
from creatures import vampire
from environments import square_room
from combat_handler import CombatHandler
from utils.agent_utils import calc_win_percentage

import curses
import sys
import traceback


def main():
    """
    Todo: Provide main documentation/overview
    """
    n_iters = int(1e6)
    # Might need move this:
    #   Initialize Q so that you can persist across different runs/episodes
    leotris.player.strategy.initialize_q(leotris)
    vampire.player.strategy.initialize_q(vampire)

    winner_list = []

    # Try to obtain console for visualization
    try:
        console = curses.initscr()
    except Exception as e:
        print(e)
        console = None

    # Attempt
    try:
        for i in range(n_iters):
            leotris.full_heal()
            vampire.full_heal()
            combat_handler = CombatHandler(
                environment=square_room,
                combatants=[leotris, vampire],
                console=console
            )
            winner = combat_handler.run()
            winner_list.append(winner)
            if (i + 1) % 10 == 0:
                win_percentages = calc_win_percentage(winner_list, [leotris, vampire])
                exploration = leotris.player.strategy.policy.get_epsilon(leotris.player.strategy.t)
                print("Win percentages: {} ({})".format(win_percentages, exploration))
                winner_list = []

    except Exception as e:
        print(e)
        traceback.print_exc(file=sys.stdout)

    if console:
        curses.endwin()


if __name__ == "__main__":
    main()
