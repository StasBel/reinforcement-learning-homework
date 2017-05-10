import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from lioncow import LionCowGame
from sarsa import SARSA
from q_learning import QLearning
from monte_carlo import MonteCarlo


def test_plot(filename):
    n, m = 10, 10
    k = 15
    algos = SARSA, QLearning, MonteCarlo
    cowns = 1, 2, 3
    for Algo, cown in product(algos, cowns):
        game = LionCowGame(n, m, cown)
        x = np.linspace(5, 3000, k).astype(np.int)
        y = np.empty(k)
        for i, iter_num in enumerate(x):
            algo = Algo(game, iter_num=iter_num)
            rewards = algo.find_op()
            y[i] = sum(rewards) / len(rewards)
        plt.plot(x, y, label="{}_{}".format(Algo.__name__, cown))
    plt.legend()
    plt.savefig(filename)


if __name__ == '__main__':
    test_plot("av_reward_plot.png")
