import gym

import utils
import main_utils
import func_utils
import cartpolev1 as cp

import numpy as np

from lake_envs import *

env = gym.make("Deterministic-8x8-FrozenLake-v0")

n_episodes = int(input("Num Episodes: "))


def x(s, a):

    # def x(s, a):
    #     vector = np.zeros((env.nA*env.nS,))
    #     vector[s*env.nA + a] = 1
    #     return vector

    # w = np.zeros((env.nA*env.nS,))


pi, w_trained = utils.monte_carlo(
    env, x, w, func_utils.policy, func_utils.monte_carlo_linear_update, n_episodes, debug=True)

input("Press to continue")

main_utils.run_environment(env, pi, 5, debug=True, render=True)
