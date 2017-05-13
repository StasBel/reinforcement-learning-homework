from q_learning import QLearning
from lioncow import LionCowGame, State, Reward, Outcome, Action
from collections import defaultdict
from random import sample, choice, choices


class DynaQ(QLearning):
    def __init__(self, *args, **kwargs):
        n = kwargs.pop("n", None)
        super().__init__(*args, **kwargs)
        self.n = n if n is not None else self.game.maxa or 100

    def find_op(self):
        Model = dict()
        steps_per_rnd = []
        for _ in range(self.iter_num):
            rnd = self.game.new_round()
            moves = set()
            steps = 0
            while True:
                steps += 1
                s = rnd.state
                a = self.eps_greedy(s)
                moves.add((s, a))
                r = rnd.do(a)
                is_over = r.outcome == Outcome.OVER
                sp = rnd.state
                self.Q[(s, a)] += self.alpha * self.q_recalc(s, a, r, sp)
                Model[(s, a)] = (r, sp)
                for _ in range(self.n):
                    s, a = sample(moves, 1)[0]
                    r, sp = Model[(s, a)]
                    self.Q[(s, a)] += self.alpha * self.q_recalc(s, a, r, sp)
                if is_over:
                    break
            steps_per_rnd.append(steps)
        return steps_per_rnd


if __name__ == '__main__':
    game = LionCowGame(10, 10, 2, maxa=500, is_stohastic=True)
    print(game.cows)
    algo = DynaQ(game)
    print(algo.find_op())
    algo.print_one_round()
