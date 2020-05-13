import numpy as np
from collections import defaultdict
import time


def epsilon_greedy(q_k, env, N_total):
    """
    From Q values generated, return a stochastic epsilon greedy policy
    """
    def pi(s):
        a = env.action_space.sample()
        epsilon = 1.0/(float(N_total[s][a])+1)
        best = max(q_k[s], key=q_k[s].get) if len(q_k[s]) > 0 else a
        return np.random.choice([a, best], p=[epsilon, 1-epsilon])

    return pi


def monte_carlo_control(pi, env, n, gamma=1.0, render=False):
    """
    This takes a policy and performs the monte carlo update
    """
    q_pi = defaultdict(lambda: defaultdict(float))
    N_total = defaultdict(lambda: defaultdict(int))

    for i in range(n):
        if i % 100 == 0:
            print("Finished {} episodes".format(i))
        G, N = monte_carlo_episode(pi, env, gamma, render)
        q_pi = monte_carlo_step(q_pi, N, G)
        for s in N:
            for a in N[s]:
                N_total[s][a] += N[s][a]
        pi = epsilon_greedy(q_pi, env, N_total)

    return q_pi, pi


def monte_carlo_episode(pi, env, gamma=1.0, render=False):
    """
    From the given policy, this takes a sample of the environment's State-Action pairs
    """
    G, k = 0, 0
    N = defaultdict(lambda: defaultdict(int))

    done = False

    s = env.reset()
    while not done and k < 500:
        if render:
            env.render()
            time.sleep(0.25)
        a = pi(s)
        s_prime, r, done, _ = env.step(a)
        G += gamma**k * r
        N[s][a] += 1
        k += 1
        s = s_prime

    return G, N


def monte_carlo_step(q_k, N, G, alpha=0.01):
    """
    This calculates a discounted sum of rewards after the end of the episode from the starting state.
    This sum is used to update the Q value for that state if there is an error between the current
    understanding and the sample.

    Q(S,A) <- Q(S,A) + alpha*(G_t - Q(S,A))
    """
    q_pi = q_k.copy()

    for s in N:
        for a in N[s]:
            if N[s][a] > 0:
                q_pi[s][a] += (1/N[s][a])*(G-q_pi[s][a])

    return q_pi


def initial_pi(env):
    """
    This returns an arbitrary pi for a given environment
    """
    return lambda s: env.action_space.sample()
