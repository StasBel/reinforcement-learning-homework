import matplotlib.pyplot as plt
import numpy as np
from random import random as rnd
from blackjack import BlackjackRound, State, Action, Outcome
from collections import defaultdict
from itertools import product


class OnPolicyMonteCarlo:
    def __init__(self, eps=1e-2, iter_num=10000):
        self.eps = eps
        self.iter_num = iter_num
        self.Q = defaultdict(float)
        self.returns = defaultdict(list)
        self.pi = defaultdict(float)
        states = (State(*p) for p in product(range(12), range(2), range(11)))
        for s, a in product(states, (Action.HIT,)):
            self.pi[(s, a)] = 1
            self.Q[(s, a)] = 1
        states = (State(*p) for p in product(range(12, 22), range(2), range(11)))
        for s, a in product(states, (Action.HIT, Action.STICK)):
            self.pi[(s, a)] = .5

    def find_op(self):
        wins, ties, loses = 0, 0, 0
        for _ in range(self.iter_num):
            blackjack_round = BlackjackRound()
            sa_log, s_log = {}, set()
            while True:
                state = blackjack_round.state
                action = Action.HIT if rnd() < self.pi[(state, Action.HIT)] else Action.STICK
                outcome = blackjack_round.do(action)
                if (state, action) not in sa_log:
                    sa_log[(state, action)] = outcome.value[1]
                if state not in s_log:
                    s_log.add(state)
                if outcome != Outcome.NOTHING:
                    if outcome == Outcome.WIN:
                        wins += 1
                    elif outcome == Outcome.TIE:
                        ties += 1
                    elif outcome == Outcome.LOSE:
                        loses += 1
                    break

            for p, r in sa_log.items():
                self.returns[p].append(r)
                self.Q[p] = sum(self.returns[p]) / len(self.returns[p])

            for s in s_log:
                astar = Action.HIT if self.Q[(s, Action.HIT)] > self.Q[(s, Action.STICK)] else Action.STICK
                for a in (Action.HIT, Action.STICK):
                    self.pi[(s, a)] = 1 - self.eps + self.eps / 2 if a == astar else self.eps / 2
        return wins, ties, loses


def test_conv(n, filename):
    x = np.linspace(5, 10000, n).astype(np.int)
    y = np.empty(n)
    for i, iter_num in enumerate(x):
        algo = OnPolicyMonteCarlo(iter_num=iter_num)
        wins, ties, loses = algo.find_op()
        y[i] = wins / (wins + ties + loses)
    plt.plot(x, y)
    plt.savefig(filename)


if __name__ == '__main__':
    test_conv(30, "iternum_conv_plot.png")
