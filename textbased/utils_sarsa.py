import utils
from collections import defaultdict
import numpy as np
import time


def sarsa(env, pi, nE, alpha=0.01, gamma=1.0, lamb=0.2, min_epsilon=0.01, debug=False, render=False):
    """Calculates Q using the on-policy SARSA method
    """
    q_pi = defaultdict(lambda: defaultdict(float))
    E = defaultdict(lambda: defaultdict(float))

    for e in range(nE):
        done = False
        s = env.reset()
        a = pi(s)
        epsilon = (1.0 - min_epsilon) * \
            (1.0 - float(e) / float(nE)) + min_epsilon

        if debug:
            if e % 1000 == 0:
                print("Completed {} episodes".format(e))
                print("Epsilon: {}".format(epsilon))

        while not done:
            if render:
                env.render()
                time.sleep(0.25)
            s_prime, r, done, _ = env.step(a)
            a_prime = pi(s_prime)

            delta = r + q_pi[s_prime][a_prime] - q_pi[s][a]
            E[s][a] += 1.0

            for s in q_pi:
                for a in q_pi[s]:
                    q_pi[s][a] += alpha * delta * E[s][a]
                    E[s][a] *= gamma*lamb

            s = s_prime
            a = a_prime
            pi = utils.epsilon_greedy(q_pi, env, epsilon)

    return q_pi, pi
