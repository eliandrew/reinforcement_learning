import numpy as np


def greedy_policy(P, V):
    """Returns a new policy by action greedily on the given
    value function V.

    Policy is a dict: (state) => action
    """
    return {s: np.argmax([action_value(P, s, a, V) for a in P[s]]) for s in P}


def deterministic_policy_to_stochastic(pi):
    return lambda a, s: 1.0 if a == pi[s] else 0.0


def policy_iteration(P, pi, gamma=1.0, delta=0.001):
    """Calculates pi_opt by iteratively improving on the initial policy.

    Returns (pi_opt, v_opt)
    """

    k = 0
    v_opt = policy_evaluation(P, pi, gamma, delta)

    max_delta = delta
    while max_delta >= delta:
        pi = deterministic_policy_to_stochastic(greedy_policy(P, v_opt))
        v_pi = policy_evaluation(P, pi, gamma, delta)
        max_delta = max([abs(v_opt[s] - v_pi[s]) for s in P])
        v_opt = v_pi
        k += 1
    pi_opt = pi

    print("Completed policy iteration in {} timesteps.".format(k))

    return pi_opt, v_opt


def policy_evaluation(P, pi, gamma=1.0, delta=0.001):
    """Calculates v_pi for the given pi

    $v_{k+1}(s) = sum_{a in A} pi(a | s) * sum_{s' in S} P_{ss'}^a (R_s^a + gamma * v_k(s'))$

    """
    k = 0
    v_pi = {s: 0.0 for s in P}

    max_delta = delta

    while max_delta >= delta:
        v_k = {s: sum([pi(a, s) * action_value(P, s, a, v_pi, gamma)
                       for a in P[s]]) for s in P}
        max_delta = max([abs(v_pi[s] - v_k[s]) for s in P])
        v_pi = v_k
        k += 1

    print("Completed policy evaluation in {} timesteps.".format(k))

    return v_pi


def value_iteration(P, gamma=1.0, delta=0.001):
    """Performs value iteration on the given MDP.

    Until convergence, update each state using the value iteration
    step.
    """
    k = 0
    v_opt = {s: 0.0 for s in P}

    max_delta = delta

    while max_delta >= delta:
        v_k = {s: value_iteration_step(P, s, v_opt, gamma) for s in P}
        max_delta = max([abs(v_k[s] - v_opt[s]) for s in P])
        v_opt = v_k
        k += 1

    print("Finished value iteration in {} time steps.".format(k))

    return v_opt


def value_iteration_step(P, s, v_k, gamma=1.0):
    """Performs a single step of the value iteration algorithm.

    $v_{k+1}(s) = max_{a in A}(sum_{s' in S}P_{ss'}^a(R_a^s + gamma * v_k(s')))$
    """

    return max([action_value(P, s, a, v_k, gamma) for a in P[s]])


def action_value(P, s, a, v_k, gamma=1.0):
    """Calculates the value of the given action.

    $sum_{s' in S}P_{ss'}^a(R_a^s + gamma * v_k(s'))$
    """

    return sum([prob * (r + gamma * v_k[s_prime]) for (prob, s_prime, r, done) in P[s][a]])


def display_values(v, n, m):
    """Displays the values in v on an nxm grid.
    """
    grid = np.array([v[s] for s in v]).reshape(n, m)
    print(grid)
