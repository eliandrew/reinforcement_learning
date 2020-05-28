import utils

import numpy as np


def linear(s, a, x, w):
    result = np.dot(x(s, a), w)
    return result


def tiling(o, n, s, a, x, w):
    result = np.dot(x(o, s, a, n), w)
    return result


def monte_carlo_linear_update(o, n, x, w, G, alpha=0.01, debug=False):
    w_next = np.copy(w)
    for (s, a, g) in G:
        delta = g - tiling(o, n, s, a, x, w_next)
        # print("Delta: {}, X: {}".format(delta, x(o, s, a, n)))

        w_change = []
        for x_ in x(o, s, a, n):
            w_change.append(alpha * delta * x_)
        w_next += w_change

    if debug:
        print("w: {}\nw_next: {}".format(w, w_next))
    return w_next


def policy(o, nT, x, w, nA, e, nE, debug=False):
    min_epsilon = 0.01
    epsilon = (1.0 - min_epsilon) * (1 - float(e) / float(nE)) + min_epsilon

    # if debug:
    #     print("Epsilon: {}".format(epsilon))

    def pi(s):
        greedy_action = np.argmax([tiling(o, nT, s, a, x, w)
                                   for a in range(nA)])
        random_action = np.random.choice(range(nA))
        return np.random.choice([greedy_action, random_action], p=[1-epsilon, epsilon])

    return pi
