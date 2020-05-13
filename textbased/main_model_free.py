import gym
import utils_model_free
import utils
import utils_model
import time
import utils_Q

from lake_envs import *

env_selection = input(
    "Choose environment:\n(1)Taxi\n(2)FrozenLake 4x4 Deterministic\n(3)FrozenLake 8x8 Deterministic\n(4)FrozenLake 4x4 Stochastic\n(5)FrozenLake 8x8 Stochastic\n")

name = "Taxi-v3"
n = 100
m = 5

if int(env_selection) == 1:
    name = "Taxi-v3"
    n = 100
    m = 5

elif int(env_selection) == 2:
    name = "Deterministic-4x4-FrozenLake-v0"
    n = 4
    m = 4

elif int(env_selection) == 3:
    name = "Deterministic-8x8-FrozenLake-v0"
    n = 8
    m = 8

elif int(env_selection) == 4:
    name = "Stochastic-4x4-FrozenLake-v0"
    n = 4
    m = 4

elif int(env_selection) == 5:
    name = "Stochastic-8x8-FrozenLake-v0"
    n = 8
    m = 8

env = gym.make(name)

total_reward = 0

n_episodes = input("Num Episodes: ")
n_episodes_watch = input("Num of episodes to watch: ")
should_render = input("Render (y/n): ")
should_render = should_render == "y"

pi = utils_model_free.initial_pi(env)

# q_sarsa, pi_sarsa = utils_model_free.sarsa(
#     env, pi, int(n_episodes), gamma=0.9, debug=True)

# q_carlo, pi_carlo = utils_model_free.monte_carlo_control(
#     pi, env, int(n_episodes), gamma=0.9, debug=True, render=False)

# v_pi = utils_model.value_iteration(env.P, 0.9)

# q_v_pi = utils.state_values_to_action_values(v_pi, env)

q_q, pi_q = utils_Q.Q_learning(env, int(n_episodes), debug=True, render=False)

# print("Q: {}\n".format(q_carlo))
pi_opt = pi_q

# for s in q_sarsa:
#     print("SARSA: {}, MC: {}, VI: {}\n".format(sorted(q_sarsa[s], key=q_sarsa[s].get), sorted(
#         q_carlo[s], key=q_carlo[s].get), sorted(q_v_pi[s], key=q_v_pi[s].get)))

input("Press enter to start simulation\n")

for e in range(int(n_episodes_watch)):
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
    int(n_episodes_watch), total_reward / float(n_episodes_watch), total_reward))
