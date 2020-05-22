import numpy as np

from collections import defaultdict


def stochastic_policy(env, weights=None):
    return lambda s: np.random.choice(range(env.action_space.n), p=weights)


def monte_carlo_sample(env, pi, gamma=0.9):
    done = False
    s = env.reset()
    G = []
    t = 0
    while not done:
        a = pi(s)
        s_prime, r, done, _ = env.step(a)
        G = [(s, a, r_prev + r * pow(gamma, t-k))
             for k, (s, a, r_prev) in enumerate(G)]
        G.append((s, a, r))
        s = s_prime
        t += 1
    return G


def monte_carlo(env, pi_from_episode, update, nE, gamma=0.9):
    for e in range(nE):
        pi = pi_from_episode(e)
        G = monte_carlo_sample(env, pi, gamma)
        update(G)
