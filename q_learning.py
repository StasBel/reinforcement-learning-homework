from lioncow import LionCowGame, State, Reward, Outcome, Action
from collections import defaultdict
from random import choice, choices


class QLearning:
    def __init__(self, game, *, eps=1e-2, alpha=0.8, gamma=0.5, iter_num=100):
        self.game = game
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.iter_num = iter_num
        self.Q = defaultdict(float)

    def eps_greedy(self, s):
        actions = list(Action)
        total = len(actions)
        qs = [self.Q[(s, a)] for a in actions]
        max_score = max(qs)
        possible_best = [a for a, q in zip(actions, qs) if q == max_score]
        best = choice(possible_best)
        weights = [1 - self.eps + self.eps / total if a == best else self.eps / total for a in actions]
        return choices(actions, weights)[0]

    def q_recalc(self, s, a, r, sp):
        mx = max(self.Q[(sp, ap)] for ap in Action)
        return r.points + self.gamma * mx - self.Q[(s, a)]

    def find_op(self):
        steps_per_rnd = []
        for _ in range(self.iter_num):
            rnd = self.game.new_round()
            steps = 0
            while True:
                steps += 1
                s = rnd.state
                a = self.eps_greedy(s)
                r = rnd.do(a)
                sp = rnd.state
                self.Q[(s, a)] += self.alpha * self.q_recalc(s, a, r, sp)
                if r.outcome == Outcome.OVER:
                    break
            steps_per_rnd.append(steps)
        return steps_per_rnd

    def print_one_round(self):
        rnd = self.game.new_round()
        while True:
            s = rnd.state
            a = self.eps_greedy(s)
            r = rnd.do(a)
            print(a, " ", r.points)
            if r.outcome == Outcome.OVER:
                break


if __name__ == '__main__':
    game = LionCowGame(10, 10, 2, maxa=500, is_stohastic=True)
    print(game.cows)
    algo = QLearning(game)
    print(algo.find_op())
    algo.print_one_round()
