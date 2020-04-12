from creatures import leotris
from creatures import vampire
from environments import square_room
from combat_handler import CombatHandler
from utils.agent_utils import calc_win_percentage

import curses
import sys
import traceback
import dill


def report_win_percentages(winner_list, num_games, combatants):
    """
    :return: None
    """
    win_percentages = calc_win_percentage(winner_list[-num_games:], combatants)
    print("Win percentages: {}".format(win_percentages))


def main():
    """
    Todo: Provide main documentation/overview
    """
    n_iters = int(1e6)
    combatants = [leotris, vampire]
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
            combat_handler = CombatHandler(
                environment=square_room,
                combatants=combatants,
                console=console
            )
            winner = combat_handler.run()
            winner_list.append(winner)

            if (i + 1) % 10 == 0:
                report_win_percentages(winner_list=winner_list, num_games=10, comatants=combatants)

            # Save tabular Q
            if (i + 1) % 100 == 0:
                dill.dump(leotris, open("results/leotris_Q_tabular.pickle", "wb"))
                dill.dump(winner_list, open("results/winner_list_Q_tabular.pickle", "wb"))

    except Exception as e:
        print(e)
        traceback.print_exc(file=sys.stdout)

    if console:
        curses.endwin()


if __name__ == "__main__":
    main()
