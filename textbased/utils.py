import numpy as np
from collections import defaultdict


def epsilon_greedy(q_k, env, epsilon):
    """
    From Q values generated, return a stochastic epsilon greedy policy
    """
    def pi(s):
        a = env.action_space.sample()
        best = max(q_k[s], key=q_k[s].get) if len(q_k[s]) > 0 else a
        return np.random.choice([a, best], p=[epsilon, 1-epsilon])

    return pi


def display_values(v, n, m):
    """Displays the values in v on an nxm grid.
    """
    grid = np.array([v[s] for s in v]).reshape(n, m)
    print(grid)


def state_values_to_action_values(v_pi, env, gamma=1.0):
    """
    This method takes an optimal value function and tranlates it into a 
    state value function for comparing model_free methods

    q_pi = R(s,a) + gamma*sum(p(s_prime,a)*v_pi(s_prime))
    """
    q_pi = defaultdict(lambda: defaultdict(float))

    for s in env.P:
        for a in env.P[s]:
            q_pi[s][a] = sum([prob * (r + gamma*v_pi[s_prime])
                              for prob, s_prime, r, _ in env.P[s][a]])

    return q_pi


def combine_nested_dicts(a, b):
    """Combines the 2D dicts of floats a and b by adding together values for
    each key.
    """
    result = a.copy()
    for outer_key in b:
        for inner_key in b[outer_key]:
            result[outer_key][inner_key] += b[outer_key][inner_key]

    # result = defaultdict(lambda: defaultdict(float))
    # for outer_key, inner_dict in a.items():
    #     for inner_key, value in inner_dict.items():
    #         result[outer_key][inner_key] += value + b[outer_key][inner_key]
    # for outer_key, inner_dict in b.items():
    #     if outer_key not in result:
    #         for inner_key, value in inner_dict.items():
    #             if inner_key not in result[outer_key]:
    #                 result[outer_key][inner_key] += value
    return result


def initial_pi(env):
    """
    This returns an arbitrary pi for a given environment
    """
    return lambda s: env.action_space.sample()
