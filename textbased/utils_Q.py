import numpy as np
from collections import Counter, defaultdict
import utils
import time


def q_learning(env, nE, min_epsilon=0.01, alpha=0.01, gamma=0.9, debug=False, render=False):
    """
    This method updates the state action values using the Q-learning algorithm

    Q(s,a) <- Q(s,a) + alpha*(r + gamma*Q(s_prime, pi_target(s_prime))-Q(s,a))))
    """

    q_pi = defaultdict(lambda: defaultdict(float))

    for e in range(nE):

        s = env.reset()
        epsilon = (1.0-min_epsilon) * \
            (1.0-float(e)/float(nE)) + min_epsilon
        done = False

        if debug:
            if e % 1000 == 0:
                print("Completed {} episodes".format(e))
                print("Epsilon {}".format(epsilon))

        while not done:
            if render:
                env.render()
                time.sleep(0.25)
            a = utils.epsilon_greedy(q_pi, env, epsilon)(s)
            s_prime, r, done, _ = env.step(a)
            a_target = utils.epsilon_greedy(q_pi, env, 0)(s_prime)

            q_pi[s][a] += alpha*(r + gamma*q_pi[s_prime][a_target]-q_pi[s][a])
            s = s_prime

    return q_pi, utils.epsilon_greedy(q_pi, env, min_epsilon)


def double_q_learning(env, nE, alpha=0.01, gamma=0.9, min_epsilon=0.01, debug=False):
    """Learns Q use the double-q learning algorithm:

     choose A using Q_1 + Q_2

     with p=.5 make update to Q_1 using Q_2:

         Q_1(S, A) = Q_1(S, A) + alpha(r + gamma * Q_2(S', pi_t(S')) - Q_1(S, A))

     with p=.5 make update to Q_2 using Q_1:

         Q_2(S, A) = Q_2(S, A) + alpha(r + gamma * Q_1(S', pi_t(S')) - Q_2(S, A))
     """

    q_1 = defaultdict(lambda: defaultdict(float))
    q_2 = defaultdict(lambda: defaultdict(float))
    for e in range(nE):

        epsilon = (1.0 - min_epsilon) * \
            (1.0 - float(e)/float(nE)) + min_epsilon

        if debug:
            if e % 1000 == 0:
                print("Completed {} episodes".format(e))
                print("Epsilon: {}".format(epsilon))
        done = False
        s = env.reset()

        while not done:
            a = utils.epsilon_greedy({s:
                                      Counter(q_1[s]) + Counter(q_2[s])}, env, epsilon)(s)
            s_prime, r, done, _ = env.step(a)

            if np.random.rand() < 0.5:
                a_prime = utils.epsilon_greedy(q_2, env, epsilon=0)(s_prime)
                q_1[s][a] += alpha * \
                    (r + gamma * q_2[s_prime][a_prime] - q_1[s][a])
            else:
                a_prime = utils.epsilon_greedy(q_1, env, epsilon=0)(s_prime)
                q_2[s][a] += alpha * \
                    (r + gamma * q_1[s_prime][a_prime] - q_2[s][a])

            s = s_prime

    q_pi = utils.combine_nested_dicts(q_1, q_2)
    return q_pi, utils.epsilon_greedy(q_pi, env, epsilon=min_epsilon)
