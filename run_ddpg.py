import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

from ddpg.ddpg_agent import DdpgAgent


def main(seed=1):
    n_episode = 1001
    max_t = 1000

    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    env.seed(seed)
    env = gym.wrappers.Monitor(env, 'result/ddpg/video', force=True)
    agent = DdpgAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        gamma=0.99,
        random_seed=seed
    )

    scores = []
    moving_scores = []
    scores_window = deque(maxlen=100)
    start_time = time.time()
    total_steps = 0

    for i_episode in range(1, n_episode+1):
        state = env.reset()
        score = 0
        for _ in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            total_steps += 1
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        moving_score = np.mean(scores_window)
        moving_scores.append(moving_score)
        steps_per_sec = total_steps / (time.time() - start_time)
        if i_episode % 100 == 0:
            end = '\n'
        else:
            end = ''
        print(f'\rEpisode {i_episode}\tAverage Score: {moving_score:.2f}\tSteps/sec: {steps_per_sec:.2f}', end=end)

    plot_scores(env_name, [scores, moving_scores], 'result/ddpg/log.png')
    agent.save_network('result/ddpg')


def plot_scores(env_name, scores_list, filename):
    n = np.arange(len(scores_list[0]))
    for scores in scores_list:
        plt.plot(n, scores)
    plt.title(env_name)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    main(0)
