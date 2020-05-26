import gym
import numpy as np
from cartpolev1 import *
env = gym.make("CartPole-v1")
o = env.observation_space
a = env.action_space.sample()
n = 10
x = course_code_x(o, o.sample(), a, n)
print("X: ", x)
