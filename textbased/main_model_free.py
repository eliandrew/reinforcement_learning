import gym
import utils
import utils_monte_carlo
import utils_sarsa
import utils_model
import time
import utils_q

from lake_envs import *

env_selection = int(input(
    "Choose environment:\n(1)Taxi\n(2)FrozenLake 4x4 Deterministic\n(3)FrozenLake 8x8 Deterministic\n(4)FrozenLake 4x4 Stochastic\n(5)FrozenLake 8x8 Stochastic\n"))

name = "Taxi-v3"
n = 100
m = 5

if env_selection == 1:
    name = "Taxi-v3"
    n = 100
    m = 5

elif env_selection == 2:
    name = "Deterministic-4x4-FrozenLake-v0"
    n = 4
    m = 4

elif env_selection == 3:
    name = "Deterministic-8x8-FrozenLake-v0"
    n = 8
    m = 8

elif env_selection == 4:
    name = "Stochastic-4x4-FrozenLake-v0"
    n = 4
    m = 4

elif env_selection == 5:
    name = "Stochastic-8x8-FrozenLake-v0"
    n = 8
    m = 8

env = gym.make(name)

total_reward = 0

min_epsilon = float(input("Min Epsilon: "))
gamma = float(input("Gamma: "))
lamb = float(input("Lambda: "))
n_episodes = int(input("Num Episodes: "))
n_episodes_watch = int(input("Num of episodes to watch: "))
should_render = "y" == input("Render (y/n): ")
should_debug = "y" == input("Debug (y/n): ")

algo = int(input(
    "Choose algorithm:\n(1)Monte-Carlo\n(2)SARSA\n(3)Q-learning\n(4)Double Q-learning\n(5)Value Iteration\n"))

pi = utils.initial_pi(env)
v_pi = utils_model.value_iteration(env.P, gamma=gamma)
q_v_pi = utils.state_values_to_action_values(v_pi, env)

pi_opt = utils.epsilon_greedy(q_v_pi, env, epsilon=min_epsilon)

if algo == 1:
    q_pi, pi_opt = utils_monte_carlo.monte_carlo_control(
        pi, env, n_episodes, gamma=gamma, min_epsilon=min_epsilon, debug=should_debug)

elif algo == 2:
    q_pi, pi_opt = utils_sarsa.sarsa(
        env, pi, n_episodes, gamma=gamma, min_epsilon=min_epsilon, lamb=lamb, debug=should_debug)

elif algo == 3:
    q_pi, pi_opt = utils_q.q_learning(
        env, n_episodes, min_epsilon=min_epsilon, gamma=gamma, lamb=lamb, debug=should_debug)

elif algo == 4:
    q_pi, pi_opt = utils_q.double_q_learning(
        env, n_episodes, gamma=gamma, min_epsilon=min_epsilon, lamb=lamb, debug=should_debug)

for s in q_pi:
    print("Algo: {}, VI: {}\n".format(
        sorted(q_pi[s], key=q_pi[s].get), sorted(q_v_pi[s], key=q_v_pi[s].get)))

input("Press enter to start simulation\n")

for e in range(n_episodes_watch):
    episode_reward = 0
    t = 0
    done = False
    s = env.reset()
    while not done:
        if should_render:
            env.render()
            time.sleep(0.25)
        (s_prime, r, done, _) = env.step(pi_opt(s))
        episode_reward += r
        s = s_prime
        t += 1

    total_reward += episode_reward
    print("Completed episode {} in {} timesteps with {} reward".format(
        e, t, episode_reward))

print("Completed {} episodes with {} average reward and {} total reward".format(
    n_episodes_watch, total_reward / float(n_episodes_watch), total_reward))
