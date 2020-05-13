import numpy as np
from collections import defaultdict
import time


def epsilon_greedy(q_k, env, epsilon):
    """
    From Q values generated, return a stochastic epsilon greedy policy
    """
    def pi(s):
        a = env.action_space.sample()
        best = max(q_k[s], key=q_k[s].get) if len(q_k[s]) > 0 else a
        return np.random.choice([a, best], p=[epsilon, 1-epsilon])

    return pi


def sarsa(env, pi, nE, alpha=0.01, gamma=1.0, epsilon_0=1.0, debug=False, render=False):
    """Calculates Q using the on-policy SARSA method
    """
    q_pi = defaultdict(lambda: defaultdict(float))
    min_epsilon = 0.01
    for e in range(nE):
        if debug:
            if e % 1000 == 0:
                print("Completed {} episodes".format(e))

        done = False
        s = env.reset()
        a = pi(s)
        epsilon = (epsilon_0 - min_epsilon) * \
            (1.0 - float(e + 1) / float(nE)) + min_epsilon
        while not done:
            s_prime, r, done, _ = env.step(a)
            a_prime = pi(s_prime)

            q_pi[s][a] += alpha * \
                (r + gamma * q_pi[s_prime][a_prime] - q_pi[s][a])

            s = s_prime
            a = a_prime
            pi = epsilon_greedy(q_pi, env, epsilon)

    return q_pi, pi


def monte_carlo_control(pi, env, n, gamma=1.0, epsilon_0=1.0, debug=False, render=False):
    """
    This takes a policy and performs the monte carlo update
    """
    q_pi = defaultdict(lambda: defaultdict(float))
    min_epsilon = 0.05
    for i in range(n):
        if debug:
            if i % 1000 == 0:
                print("Finished {} episodes".format(i))
        epsilon = (epsilon_0 - min_epsilon) * \
            (1 - float(i + 1) / float(n)) + min_epsilon
        G, N = monte_carlo_episode(pi, env, gamma, render)
        q_pi = monte_carlo_step(q_pi, N, G)
        pi = epsilon_greedy(q_pi, env, epsilon)

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
                q_pi[s][a] += alpha*(G-q_pi[s][a])

    return q_pi


def initial_pi(env):
    """
    This returns an arbitrary pi for a given environment
    """
    return lambda s: env.action_space.sample()
