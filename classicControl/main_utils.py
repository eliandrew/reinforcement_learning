import time


def run_environment(env, pi, obs_to_state, nE, debug=False, render=False):

    total_reward = 0

    for e in range(nE):
        reward = 0
        done = False
        o = env.reset()
        o_prev = [0] * len(o)
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            s = obs_to_state(o_prev, o)
            a = pi(s)
            o_new, r, done, _ = env.step(a)
            o_prev = o
            o = o_new
            reward += r
        total_reward += reward
        if debug:
            print("Completed episode {} with {} reward".format(e, reward))
    print("Completed {} episodes with {} total reward".format(nE, total_reward))
