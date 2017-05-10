from math import exp
from random import randint
from enum import Enum, auto
from collections import namedtuple


class Pt(namedtuple("Pt", ["x", "y"])):
    def __add__(self, other):
        return Pt(self.x + other.x, self.y + other.y)

    def __iadd__(self, other):
        return self + other


State = namedtuple("State", ["pos", "cows"])


class Action(Enum):
    LEFT = Pt(-1, 0)
    UP = Pt(0, 1)
    RIGHT = Pt(1, 0)
    DOWN = Pt(0, -1)


class Outcome(Enum):
    NOTHING = auto()
    OVER = auto()


Reward = namedtuple("Reward", ["outcome", "points"])


class Board(namedtuple("Board", ["n", "m"])):
    def is_inside(self, pt):
        return 1 <= pt.x <= self.n and 1 <= pt.y <= self.m


Cow = namedtuple("Cow", ["pos"])


class Lion:
    START_PT = Pt(1, 1)

    def __init__(self):
        self.pos = Lion.START_PT
        self.cows = set()

    def move(self, pt):
        self.pos = pt

    def pick_cow(self, cow):
        self.cows.add(cow)

    def eat_cows(self):
        self.cows.clear()


class LionCowGame:
    def __init__(self, n, m, cown, maxa=None):
        self.board = Board(n, m)
        pts = set()
        for _ in range(cown):
            while True:
                pt = Pt(randint(1, n), randint(1, m))
                if pt != Lion.START_PT and pt not in pts:
                    pts.add(pt)
                    break
        self.cows = frozenset(Cow(pt) for pt in pts)
        self.maxa = 4 * n * m if maxa is None else maxa

    def new_round(self):
        return LionCowRound(self.board, self.cows, self.maxa)


class LionCowRound:
    JUST_MOVE = 0
    PICK_COW = 50
    BRING_COWS = 500
    NO_COW_BACK_PENALTY = -50
    WRONG_MOVE_PENALTY = -10

    def __init__(self, board, cows, maxa):
        self.board = board
        self.cows = set(cows)
        self.maxa = maxa
        self.cura = 0
        self.lion = Lion()

    def _do(self, action):
        dpt = self.lion.pos + action.value
        if self.board.is_inside(dpt):
            self.lion.move(dpt)
            pcow = Cow(dpt)
            if pcow in self.cows:
                self.cows.remove(pcow)
                self.lion.pick_cow(pcow)
                points = len(self.lion.cows) * LionCowRound.PICK_COW
                return Reward(Outcome.NOTHING, points)
            else:
                if dpt == Lion.START_PT:
                    if not len(self.cows):
                        self.lion.eat_cows()
                        return Reward(Outcome.OVER, LionCowRound.BRING_COWS)
                    else:
                        return Reward(Outcome.NOTHING, LionCowRound.NO_COW_BACK_PENALTY)
                else:
                    return Reward(Outcome.NOTHING, LionCowRound.JUST_MOVE)
        else:
            return Reward(Outcome.NOTHING, LionCowRound.WRONG_MOVE_PENALTY)

    def do(self, action):
        self.cura += 1
        reward = self._do(action)
        if self.cura == self.maxa:
            reward = reward._replace(outcome=Outcome.OVER)
        return reward

    @property
    def state(self):
        return State(self.lion.pos, frozenset(self.lion.cows))
