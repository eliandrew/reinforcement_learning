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


def Q_learning(env, nE, epsilon_0=1.0, alpha=0.01, gamma=0.9, debug=False, render=False):
    """
    This method updates the state action values using the Q-learning algorithm

    Q(s,a) <- Q(s,a) + alpha*(r + gamma*Q(s_prime, pi_target(s_prime))-Q(s,a))))
    """

    q_pi = defaultdict(lambda: defaultdict(float))
    min_epsilon = 0.05

    for e in range(nE):
        if debug:
            if e % 1000 == 0:
                print("Completed {} episodes".format(e))

        s = env.reset()
        epsilon = (epsilon_0-min_epsilon) * \
            (1.0-float(e+1)/float(nE)) + min_epsilon
        done = False

        while not done:
            if render:
                env.render()
                time.sleep(0.25)
            a = epsilon_greedy(q_pi, env, epsilon)(s)
            s_prime, r, done, _ = env.step(a)
            a_target = epsilon_greedy(q_pi, env, 0)(s_prime)

            q_pi[s][a] += alpha*(r + gamma*q_pi[s_prime][a_target]-q_pi[s][a])
            s = s_prime

    return q_pi, epsilon_greedy(q_pi, env, min_epsilon)
