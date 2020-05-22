import time


def run_environment(env, pi, nE, debug=False, render=False):

    total_reward = 0

    for e in range(nE):
        reward = 0
        done = False
        s = env.reset()
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            a = pi(s)
            s_prime, r, done, _ = env.step(a)
            s = s_prime
            reward += r
        total_reward += reward
        if debug:
            print("Completed episode {} with {} reward".format(e, reward))
    print("Completed {} episodes with {} total reward".format(nE, total_reward))
