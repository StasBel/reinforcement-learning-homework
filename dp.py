from lioncow import LionCowGame, State, Reward, Outcome, Action
from collections import defaultdict


class DP:
    def __init__(self, game, *, iter_num=10000):
        self.game = game
        self.iter_num = iter_num
        self.V = defaultdict(float)

    def find_op(self):
        rewards = []
        for _ in range(self.iter_num):
            pass


if __name__ == '__main__':
    pass
