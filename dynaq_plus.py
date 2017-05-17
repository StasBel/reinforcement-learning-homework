from queue import PriorityQueue
from dynaq import DynaQ
from lioncow import LionCowGame, State, Reward, Outcome, Action
from collections import defaultdict, Counter
from random import sample, choice, choices
from functools import partial
from operator import itemgetter
from statistics import mode


class DynaQPlus(DynaQ):
    def __init__(self, *args, **kwargs):
        tetta = kwargs.pop("tetta", 1e-2)
        super().__init__(*args, **kwargs)
        self.tetta = tetta

    def find_op(self):
        Model = dict()
        steps_per_rnd = []
        for _ in range(self.iter_num):
            rnd = self.game.new_round()
            PQueue = set()
            Moves = defaultdict(partial(defaultdict, int))
            steps = 0
            while True:
                steps += 1
                s = rnd.state
                a = self.eps_greedy(s)
                r = rnd.do(a)
                is_over = r.outcome == Outcome.OVER
                sp = rnd.state
                Moves[sp][(s, a, r)] += 1
                Model[(s, a)] = (r, sp)
                P = abs(self.q_recalc(s, a, r, sp))
                if P > self.tetta:
                    PQueue.add((P, s, a))
                for _ in range(self.n):
                    if not len(PQueue):
                        break
                    maxp = max(p for p, _, _ in PQueue)
                    s, a = next((s, a) for p, s, a in PQueue if p == maxp)
                    PQueue.remove((maxp, s, a))
                    r, sp = Model[(s, a)]
                    self.Q[(s, a)] += self.alpha * self.q_recalc(s, a, r, sp)
                    if len(Moves[s]):
                        st, at, rt = max(Moves[s].items(), key=itemgetter(1))[0]
                        P = abs(self.q_recalc(st, at, rt, s))
                        if P > self.tetta:
                            PQueue.add((P, st, at))
                if is_over:
                    break
            steps_per_rnd.append(steps)
        return steps_per_rnd


if __name__ == '__main__':
    game = LionCowGame(5, 5, 2, is_stohastic=True, maxa=500, stohastic_p=0.3)
    print(game.cows)
    algo = DynaQPlus(game)
    rewards = algo.find_op()
    print(rewards)
    print(sum(reward == 500 for reward in rewards))
    algo.print_one_round()
