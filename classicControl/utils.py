import numpy as np


def stochastic_policy(env, weights=None):
    return lambda s: np.random.choice(range(env.action_space.n), p=weights)
