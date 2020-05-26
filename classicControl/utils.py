import numpy as np

from collections import defaultdict


def stochastic_policy(env, weights=None):
    return lambda s: np.random.choice(range(env.action_space.n), p=weights)


def monte_carlo_sample(env, pi, obs_to_state, gamma=0.9, debug=False):
    done = False
    o = env.reset()
    o_prev = [0] * len(o)
    G = []
    t = 0
    while not done:
        s = obs_to_state(o_prev, o)
        a = pi(s)
        o_new, r, done, _ = env.step(a)
        G = [(s, a, r_prev + r * pow(gamma, t-k))
             for k, (s, a, r_prev) in enumerate(G)]
        G.append((s, a, r))
        o_prev = o
        o = o_new
        t += 1
    # if debug:
    #     print("r: {}".format([r for (_, _, r) in G]))
    return G


def monte_carlo(env, x, w, obs_to_state, update_policy, update_weights, nE, gamma=0.9, alpha=0.01, debug=False):
    w_trained = np.copy(w)
    for e in range(nE):
        if e % 1000 == 0:
            print("Completed {} episodes".format(e))
            print("w: {}".format(w_trained))
        pi = update_policy(x, w_trained, env.action_space.n,
                           e, nE, debug=debug)
        G = monte_carlo_sample(env, pi, obs_to_state, gamma, debug=debug)
        w_trained = update_weights(x, w_trained, G, alpha)
    return pi, w_trained
