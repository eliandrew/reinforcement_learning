import gym
import utils
import time

from lake_envs import *

env_selection = input(
    "Choose environment:\n(1)Taxi\n(2)FrozenLake 4x4 Deterministic\n(3)FrozenLake 4x4 Stochastic\n(4)FrozenLake 8x8 Deterministic\n(5)FrozenLake 8x8 Stochastic\n")

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
should_render = input("Render (y/n): ")
should_render = should_render == "y"

v_opt = utils.value_iteration(env.P, gamma=0.9, delta=0.001)

print("Value Function:\n")
utils.display_values(v_opt, n, m)

pi_opt = utils.greedy_policy(env.P, v_opt)

print("Policy:\n")
pi_values = [pi_opt[s] for s in pi_opt]
utils.display_values(pi_values, n, m)

input("Press enter to start simulation\n")

for e in range(int(n_episodes)):
    episode_reward = 0
    t = 0
    done = False
    s = env.reset()
    while not done:
        if should_render:
            env.render()
            time.sleep(0.25)
        (s_prime, r, done, _) = env.step(pi_opt[s])
        episode_reward += r
        s = s_prime
        t += 1

    total_reward += episode_reward
    print("Completed episode {} in {} timesteps with {} reward".format(
        e, t, episode_reward))

print("Completed {} episodes with {} average reward and {} total reward".format(
    int(n_episodes), total_reward / float(n_episodes), total_reward))
