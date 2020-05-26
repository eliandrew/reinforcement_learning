import utils

import numpy as np


def linear(s, a, x, w):
    result = np.dot(x(s, a), w)
    return result


def monte_carlo_linear_update(x, w, G, alpha=0.01, debug=False):
    w_next = np.copy(w)
    for (s, a, g) in G:
        w_next += alpha * (g - linear(s, a, x, w_next)) * x(s, a)
    if debug:
        print("w: {}\nw_next: {}".format(w, w_next))
    return w_next


def policy(x, w, nA, e, nE, debug=False):
    min_epsilon = 0.01
    epsilon = (1.0 - min_epsilon) * \
        (1 - float(e) / float(nE)) + min_epsilon

    # if debug:
    #     print("Epsilon: {}".format(epsilon))

    def pi(s):
        greedy_action = np.argmax([linear(s, a, x, w) for a in range(nA)])
        random_action = np.random.choice(range(nA))
        return np.random.choice([greedy_action, random_action], p=[1-epsilon, epsilon])

    return pi
