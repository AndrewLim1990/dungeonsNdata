from agents import TIME_LIMIT
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


def report_win_percentages(winner_list, num_games, combatants, total_rewards, last_states, num_actions_takens):
    """
    :return: None
    """
    win_percentages = calc_win_percentage(winner_list[-num_games:], combatants)
    last_states = torch.cat(last_states).data.numpy()
    print("Win percentages: {}\t".format(win_percentages))

    results = list(zip(winner_list[-num_games:], total_rewards[-num_games:], last_states, num_actions_takens))
    results = sorted(results, key=lambda x: -x[1])

    for winner, avg_reward, last_state, num_actions_taken in results:
        print(" {}: {} ({}) \t\t{}".format(winner, avg_reward, last_state, num_actions_taken))
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
    total_rewards = []
    last_states = []
    num_actions_takens = []

    for i in range(n_iters):
        combat_handler = CombatHandler(
            environment=square_room,
            combatants=[leotris, vampire],
            time_limit=TIME_LIMIT
        )
        intialize_combatants([leotris, vampire], combat_handler=combat_handler)
        winner, total_reward, last_state, num_actions_taken = combat_handler.run()

        winner_list.append(winner)
        total_rewards.append(total_reward)
        last_states.append(last_state)
        num_actions_takens.append(num_actions_taken)

        if (i + 1) % 10 == 0:
            report_win_percentages(
                winner_list=winner_list,
                num_games=10,
                combatants=[leotris, vampire],
                total_rewards=total_rewards,
                last_states=last_states,
                num_actions_takens=num_actions_takens
            )
            last_states = []
            num_actions_takens = []

        # Save tabular Q
        if (i + 1) % 10 == 0:
            dill.dump(winner_list, open("results/winner_list_{}.pickle".format(leotris.strategy.name), "wb"))
            dill.dump(leotris.strategy.policy_net, open("results/model_{}.pickle".format(leotris.strategy.name), "wb"))
            dill.dump(total_rewards, open('results/reward_list_{}.pickle'.format(leotris.strategy.name), "wb"))


if __name__ == "__main__":
    main()
