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


def course_code_x(obs, s, a, n):
    fixed_bound = 100000
    section_lengths = [round((min(fixed_bound, obs.high[i]) -
                              max(-(fixed_bound), obs.low[i]))/n) for i in range(obs.shape[0])]

    print("Section Lengths: ", section_lengths)

    indicies = [(value % section_lengths[index]) *
                index * n for index, value in enumerate(s)]
    print("Indicies: ", indicies)

    features = [1 if i in indicies else 0 for i in range(len(s)*n+1)]
    return features


def obs_to_state(o_prev, o):
    return (o_prev, o)
