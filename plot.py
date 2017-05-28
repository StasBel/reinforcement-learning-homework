import matplotlib.pyplot as plt
import numpy as np
from sarsa import SarsaSemiGradientMountainCar


def plot_learning_curve():
    runs = 3
    episodes = 500
    num_of_tilings = 16
    pos_grid_size = 20
    vel_grid_size = 10
    eps = 1e-3
    alpha = 0.8
    gamma = 0.95

    rewards = np.zeros(episodes)
    for _ in range(runs):
        algo = SarsaSemiGradientMountainCar(num_of_tilings, pos_grid_size, vel_grid_size,
                                            eps=eps, alpha=alpha, gamma=gamma)
        rewards += algo.find_op(num_of_episodes=episodes)
    rewards /= runs
    plt.plot(rewards, color="green")
    plt.title("Learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Steps per episode")
    plt.savefig("learning_curve.png")
    # plt.show()


if __name__ == '__main__':
    plot_learning_curve()
