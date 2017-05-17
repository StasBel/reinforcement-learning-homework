import numpy as np
import matplotlib.pyplot as plt
from lioncow import LionCowGame
from q_learning import QLearning
from dynaq import DynaQ
from dynaq_plus import DynaQPlus
from itertools import chain, product
from functools import partial

games = LionCowGame(10, 10, 2), LionCowGame(10, 10, 2, maxa=500, is_stohastic=True)
nalgos = DynaQ, DynaQPlus
ns = 1, 50, 100
algos = lambda n: tuple(chain((QLearning,), (partial(a, n=n) for a, n in product(nalgos, (n,)))))
iter_num = 50


def draw_plot(game, filename, n):
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.figure()
    for Algo in algos(n):
        algo = Algo(game)
        steps = algo.find_op()
        label = "{}{}".format(algo.__class__.__name__, ", n={}".format(algo.n) if hasattr(algo, "n") else "")
        plt.plot(steps, label=label)
    plt.legend(prop={"size": 15})
    plt.savefig(filename)


if __name__ == '__main__':
    for (i, game), n in product(enumerate(games[1:]), ns):
        filename = "{}{}_plot.png".format("stohastic" if game.stoh_p > 0 else "deterministic", n)
        draw_plot(game, filename, n)
