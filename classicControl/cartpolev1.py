import numpy as np


def initialize_weights(n):
    return [np.random.uniform(-1, 1) for _ in range(n)]


def identity_x(s, a):
    return np.hstack((s[0], s[1], a))


def difference_x(s, a):
    o_prev, o = s
    return np.array(o) - np.array(o_prev)


def product_x(s, a):
    _, o = s
    return np.array([o[0] * o[1], o[2] * o[3]])


def concat_x(s, a):
    return np.hstack((identity_x(s, a), difference_x(s, a), product_x(s, a)))


def obs_to_state(o_prev, o):
    return (o_prev, o)
