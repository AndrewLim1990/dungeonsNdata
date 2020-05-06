from creatures import vampire
from creatures import leotris
from environments import square_room
from combat_handler import CombatHandler
from utils.agent_utils import calc_win_percentage

import dill
import numpy as np
import torch

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# leotris = dill.load(open("results/model_double_DQN_2.pickle", "rb"))


def report_win_percentages(winner_list, num_games, combatants, q_vals, last_states, num_actions_takens):
    """
    :return: None
    """
    win_percentages = calc_win_percentage(winner_list[-num_games:], combatants)
    q_vals = torch.tensor(q_vals).tolist()
    last_states = np.around(np.array(last_states), 2)

    print("Win percentages: {}\t{}".format(
        win_percentages,
        leotris.player.strategy.policy.get_epsilon(leotris.player.strategy.t),
    ))

    results = list(zip(winner_list[-num_games:], q_vals, last_states, num_actions_takens))
    results = sorted(results, key=lambda x: -x[1])

    for winner, q_val, last_state, num_actions_taken in results:
        print(" {}: {} ({}) \t\t{}".format(winner, round(q_val, 3), last_state, num_actions_taken))
    print("----------------------\n")


def intialize_combatants(combatants, combat_handler):
    """
    :param combatants:
    :return:
    """
    [combatant.initialize(combat_handler) for combatant in combatants]

def main():
    """
    Todo: Provide main documentation/overview
    """
    n_iters = int(1e6)

    winner_list = []
    q_vals = []
    last_states = []
    num_actions_takens = []

    for i in range(n_iters):
        combat_handler = CombatHandler(
            environment=square_room,
            combatants=[leotris, vampire]
        )
        intialize_combatants([leotris, vampire], combat_handler=combat_handler)
        winner, q_val, last_state, num_actions_taken = combat_handler.run()

        winner_list.append(winner)
        q_vals.append(q_val)
        last_states.append(last_state)
        num_actions_takens.append(num_actions_taken)

        if (i + 1) % 10 == 0:
            report_win_percentages(
                winner_list=winner_list,
                num_games=10,
                combatants=[leotris, vampire],
                q_vals=q_vals,
                last_states=last_states,
                num_actions_takens=num_actions_takens
            )
            q_vals = []
            last_states = []
            num_actions_takens = []

        # Save tabular Q
        if (i + 1) % 100 == 0:
            dill.dump(winner_list, open("results/winner_list_{}.pickle".format(leotris.player.strategy.name), "wb"))
            dill.dump(leotris, open("results/model_{}_3.pickle".format(leotris.player.strategy.name), "wb"))


if __name__ == "__main__":
    main()
