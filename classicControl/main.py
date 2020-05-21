import gym

import utils
import main_utils

env = gym.make("CartPole-v1")

rand_pi = utils.stochastic_policy(env)

main_utils.run_environment(env, rand_pi, 5, debug=True, render=True)
