import gym

import utils
import main_utils
import func_utils
import cartpolev1 as cp

import numpy as np

from lake_envs import *

env = gym.make("CartPole-v1")

n_episodes = int(input("Num Episodes: "))

# def x(s, a):
#     vector = np.zeros((env.nA*env.nS,))
#     vector[s*env.nA + a] = 1
#     return vector

# w = np.zeros((env.nA*env.nS,))

x = cp.concat_x
w = np.zeros((15,))

pi, w_trained = utils.monte_carlo(
    env, x, w, cp.obs_to_state, func_utils.policy, func_utils.monte_carlo_linear_update, n_episodes, debug=True)

input("Press to continue")

main_utils.run_environment(env, pi, cp.obs_to_state,
                           100, debug=True, render=True)
