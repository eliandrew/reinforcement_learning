import utils

import numpy as np


def linear(s, a, x, w):
    return np.dot(x(s, a), w)


def monte_carlo_linear_update(x, w, G, alpha=0.01):
    for s in G:
        for a in G[s]:
            g_t, _ = G[s][a]
            delta = g_t - linear(s, a, x, w)
            w += alpha * delta * x(s, a)


def policy(x, w, episode):


def monte_carlo_func_approx(env):
