import numpy as np
from collections import defaultdict


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
