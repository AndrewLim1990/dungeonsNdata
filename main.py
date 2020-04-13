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


def intialize_combatants(combatants):
    """
    :param combatants:
    :return:
    """
    [combatant.initialize() for combatant in combatants]


def main():
    """
    Todo: Provide main documentation/overview
    """
    n_iters = int(1e6)
    combatants = [leotris, vampire]
    intialize_combatants(combatants)

    winner_list = []

    for i in range(n_iters):
        combat_handler = CombatHandler(
            environment=square_room,
            combatants=combatants
        )
        winner = combat_handler.run()
        winner_list.append(winner)

        if (i + 1) % 10 == 0:
            report_win_percentages(winner_list=winner_list, num_games=10, comatants=combatants)

        # Save tabular Q
        if (i + 1) % 100 == 0:
            dill.dump(leotris, open("results/leotris_Q_tabular.pickle", "wb"))
            dill.dump(winner_list, open("results/winner_list_Q_tabular.pickle", "wb"))


if __name__ == "__main__":
    main()
