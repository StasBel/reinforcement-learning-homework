from random import shuffle, choice
from collections import namedtuple
from enum import Enum, auto

VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
          '8': 8, '9': 9, 'T': 10, 'J': 10, 'Q': 10, 'K': 10}
SUITS_NUM = 4
RANKS = tuple(VALUES.keys())

Card = namedtuple("Card", ["rank"])


class Hand:
    def __init__(self):
        self.cards = []

    def add(self, card):
        self.cards.append(card)

    @property
    def last_card(self):
        return self.cards[-1]

    @property
    def use_ace(self):
        value = sum(VALUES[card.rank] for card in self.cards)
        for card in self.cards:
            if card.rank == "A" and value <= 11:
                return True
        return False

    @property
    def value(self):
        value = sum(VALUES[card.rank] for card in self.cards)
        if self.use_ace:
            value += 10
        return value


class Deck:
    def __init__(self):
        self.cards = [Card(rank) for rank in RANKS * SUITS_NUM]
        shuffle(self.cards)

    def deal(self):
        return self.cards.pop()


State = namedtuple("State", ["csum", "uace", "dcard"])


class Action(Enum):
    HIT = auto()
    STICK = auto()


class Outcome(Enum):
    NOTHING = 0, 0
    WIN = 1, 1
    LOSE = 1, -1
    TIE = 1, 0


class BlackjackRound:
    def __init__(self):
        self.deck = Deck()
        self.player = Hand()
        self.dealer = Hand()
        self.player.add(self.deck.deal())
        self.dealer.add(self.deck.deal())
        self.player.add(self.deck.deal())
        self.dealer.add(self.deck.deal())

    def _hit(self):
        self.player.add(self.deck.deal())
        if self.player.value > 21:
            return Outcome.LOSE
        return Outcome.NOTHING

    def _stick(self):
        while self.dealer.value < 17:
            self.dealer.add(self.deck.deal())
        if self.dealer.value > 21:
            return Outcome.WIN
        if self.dealer.value > self.player.value:
            return Outcome.LOSE
        if self.dealer.value == self.player.value:
            return Outcome.TIE
        if self.dealer.value < self.player.value:
            return Outcome.WIN
        return Outcome.NOTHING

    def do(self, action):
        if action == Action.HIT:
            return self._hit()
        else:
            return self._stick()

    @property
    def state(self):
        return State(self.player.value,
                     int(self.player.use_ace),
                     VALUES[self.dealer.last_card.rank])
