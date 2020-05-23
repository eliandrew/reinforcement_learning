import numpy as np


def initialize_weights(n):
    return [np.random.uniform(-1, 1) for _ in range(n)]


def identity_x(s, a):
    return np.append(s, a)
