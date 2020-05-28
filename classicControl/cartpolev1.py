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
    section_lengths = [(min(fixed_bound, obs.high[i]) -
                        max(-(fixed_bound), obs.low[i]))/n for i in range(obs.shape[0])]

    sections = []
    for l in section_lengths:
        bins = []
        for i in range(n-1):
            bins.append(l*i)
        sections.append(bins)

    indicies = []
    for index, section in enumerate(sections):
        indicies.append(np.digitize(s[index], section) + index*n)

    features = [1 if i in indicies else 0 for i in range(len(s)*n+1)]
    features[-1] = a
    return features


def obs_to_state(o_prev, o):
    return (o_prev, o)


def obs_tiling(o_prev, o):
    return o
