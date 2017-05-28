import gym
import numpy as np
from tiles import tiles, IHT


class Tiling:
    def __init__(self, num_of_tilings, floats_bounds, ints_bounds):
        self.num_of_tilings = num_of_tilings
        size = num_of_tilings
        scales = []
        for l, r, s in floats_bounds:
            size *= s
            scales.append(s / (r - l))
        self._scales = scales
        for l, r in ints_bounds:
            size *= (r - l + 1)
        self.iht = IHT(size)

    @property
    def size(self):
        return self.iht.size

    def _normalize(self, floats, ints):
        nfloats = []
        for d, s in zip(floats, self._scales):
            nfloats.append(d * s)
        return nfloats, ints

    def activeTiles(self, floats, ints):
        if isinstance(ints, int):
            ints = [ints]
        if isinstance(floats, float):
            floats = [floats]
        floats, ints = self._normalize(floats, ints)
        return tiles(self.iht, self.num_of_tilings, floats, ints)


class SarsaSemiGradientMountainCar:
    def __init__(self, num_of_tilings, pos_grid_size, vel_grid_size, *, eps=1e-3, alpha=0.8, gamma=0.95):
        self.env = gym.make("MountainCar-v0")
        lp, lv = self.env.observation_space.low
        hp, hv = self.env.observation_space.high
        self.AS = list(range(self.env.action_space.n))
        la, ha = min(self.AS), max(self.AS)
        self.tiling = Tiling(num_of_tilings,
                             [(lp, hp, pos_grid_size), (lv, hv, vel_grid_size)],
                             [(la, ha)])
        self.tetta = np.zeros(self.tiling.size)
        self.eps = eps
        self.alpha = alpha / num_of_tilings
        self.gamma = gamma

    def Q(self, S, A):
        active_tiles = self.tiling.activeTiles(S, A)
        return self.tetta[active_tiles].sum()

    def dQ(self, S, A):
        active_tiles = self.tiling.activeTiles(S, A)
        dF = np.zeros(len(self.tetta))
        dF[active_tiles] = 1
        return dF

    def choose_action(self, S, eps_greedy=True):
        if eps_greedy and np.random.binomial(1, self.eps) == 1:
            return int(np.random.choice(self.AS))
        QS = [self.Q(S, A) for A in self.AS]
        mQS = max(QS)
        bAS = [A for A, Q in enumerate(QS) if Q == mQS]
        return int(np.random.choice(bAS))

    def find_op(self, *, num_of_episodes=1, do_render=False):
        steps_nums = []
        for _ in range(num_of_episodes):
            S = self.env.reset()
            A = self.choose_action(S, not do_render)
            if do_render:
                self.env.render()
            step_num = 0
            while True:
                step_num += 1
                Sp, R, done, _ = self.env.step(A)
                if do_render:
                    self.env.render()
                if done:
                    self.tetta += self.alpha * (R - self.Q(S, A)) * self.dQ(S, A)
                    break
                Ap = self.choose_action(Sp, not do_render)
                self.tetta += self.alpha * (R + self.gamma * self.Q(Sp, Ap) - self.Q(S, A)) * self.dQ(S, A)
                S, A = Sp, Ap
            steps_nums.append(step_num)
        return np.array(steps_nums)

    def draw_round(self):
        return self.find_op(do_render=True)


if __name__ == '__main__':
    algo = SarsaSemiGradientMountainCar(10, 20, 10)
    algo.find_op(num_of_episodes=500)
    print(algo.draw_round())
