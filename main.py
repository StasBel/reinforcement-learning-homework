import numpy as np


class LionCowDomain:
    """
    0 - unknown
    1 - left
    2 - up
    3 - right
    4 - down
    5 - stay
        2
        |
    1 --5-- 3
        |
        4   #0
    """

    def __init__(self, n, m, actions=6):
        self.n = n
        self.m = m
        self.actions = actions
        reward = np.zeros((n + 2, m + 2, self.actions))
        reward[n - 1, m, 3] = 100
        reward[n, m - 1, 2] = 100
        self.reward = reward

    def moves(self):
        return range(1, self.actions)


class LearningAlgorithm:
    def compute_Q(self):
        Q = np.zeros((self.domain.n + 2, self.domain.m + 2, self.domain.actions))
        views = self.V[:-2, 1:-1], self.V[1:-1, 2:], self.V[2:, 1:-1], self.V[1:-1, :-2], self.V[1:-1, 1:-1]
        for move, view in zip(self.domain.moves(), views):
            Q[1:-1, 1:-1, move] = self.domain.reward[1:-1, 1:-1, move] + self.gamma * view
        Q[self.domain.n, self.domain.m] = np.zeros(self.domain.actions)
        return Q


class PolicyIteration(LearningAlgorithm):
    def __init__(self, domain, peps=1e-5, gamma=1):
        self.domain = domain
        self.peps = peps
        self.gamma = gamma
        n, m = domain.n, domain.m
        P = np.zeros((n + 2, m + 2))
        self.P = P
        self.V = np.zeros((n + 2, m + 2))

    def eval_policy(self):
        num_iter = 0
        while True:
            num_iter += 1
            a, b = np.indices(self.P.shape)
            next_V = super(PolicyIteration, self).compute_Q()[a, b, self.P.astype(np.int)]
            delta = (abs(self.V - next_V))[1:-1, 1:-1].max()
            self.V = next_V
            if delta < self.peps:
                break
        return num_iter

    def improve_policy(self):
        num_iter = 0
        while True:
            num_iter += 1
            num_iter += self.eval_policy()
            next_P = super(PolicyIteration, self).compute_Q().argmax(axis=2)
            isStable = (self.P == next_P).all()
            self.P = next_P
            if isStable:
                break
        return num_iter


class ValueIteration(LearningAlgorithm):
    def __init__(self, domain, peps=1e-5, gamma=1):
        self.domain = domain
        self.peps = peps
        self.gamma = gamma
        n, m = domain.n, domain.m
        self.V = np.zeros((n + 2, m + 2))
        self.P = None

    def eval_policy(self):
        num_iter = 0
        while True:
            num_iter += 1
            next_V = super(ValueIteration, self).compute_Q().max(axis=2)
            delta = (abs(self.V - next_V))[1:-1, 1:-1].max()
            self.V = next_V
            if delta < self.peps:
                break
        return num_iter

    def improve_policy(self):
        num_iter = self.eval_policy()
        P = np.zeros((self.domain.n + 2, self.domain.m + 2))
        P[1:-1, 1:-1] = super(ValueIteration, self).compute_Q().argmax(axis=2)[1:-1, 1:-1]
        self.P = P
        return num_iter


PRINTER = {0: '?', 1: '←', 2: '↑', 3: '→', 4: '↓', 5: '↻'}


def nice_print(P):
    n, m = P.shape
    mtx = P[1:-1, 1:-1][:, ::-1].T
    for i in range(n - 2):
        for j in range(m - 2):
            print(PRINTER[mtx[i][j]], end=' ')
        print()


if __name__ == '__main__':
    dmn = LionCowDomain(10, 10)
    alg = PolicyIteration(dmn, gamma=0.95)
    its = alg.improve_policy()
    print("Number of iterations: {}".format(its))
    nice_print(alg.P)
