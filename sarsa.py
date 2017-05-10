from lioncow import LionCowGame, State, Reward, Outcome, Action
from collections import defaultdict
from random import choice, choices


class SARSA:
    def __init__(self, game, *, eps=1e-1, alpha=0.8, gamma=0.5, iter_num=10000):
        self.game = game
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.iter_num = iter_num
        self.Q = defaultdict(float)

    def choose_action(self, s):
        actions = list(Action)
        total = len(actions)
        qs = [self.Q[(s, a)] for a in actions]
        max_score = max(qs)
        possible_best = [a for a, q in zip(actions, qs) if q == max_score]
        best = choice(possible_best)
        weights = [1 - self.eps + self.eps / total if a == best else self.eps / total for a in actions]
        return choices(actions, weights)[0]

    def find_op(self):
        rewards = []
        for _ in range(self.iter_num):
            rnd = self.game.new_round()
            treward = 0
            while True:
                s = rnd.state
                a = self.choose_action(s)
                r = rnd.do(a)
                sp = rnd.state
                ap = self.choose_action(sp)
                self.Q[(s, a)] += self.alpha * (r.points + self.gamma * self.Q[(sp, ap)] - self.Q[(s, a)])
                treward += r.points
                if r.outcome == Outcome.OVER:
                    break
            rewards.append(treward)
        return rewards

    def print_one_round(self):
        rnd = self.game.new_round()
        while True:
            s = rnd.state
            a = self.choose_action(s)
            r = rnd.do(a)
            print(a, " ", r.points)
            if r.outcome == Outcome.OVER:
                break


if __name__ == '__main__':
    game = LionCowGame(5, 5, 3)
    print(game.cows)
    sarsa = SARSA(game)
    rewards = sarsa.find_op()
    sarsa.print_one_round()
