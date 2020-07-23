import gym
import numpy as np
from dqn.dqn_agent import DqnAgent
from collections import deque

import matplotlib.pyplot as plt
import time


def main(seed=1):
    n_episodes = 1001
    max_t = 1000
    eps = 1
    end_eps = 0.01
    eps_decay = 0.995

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env.seed(seed)
    env = gym.wrappers.Monitor(env, 'result/video', force=True)
    agent = DqnAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        soft_target_update=True,
        double=True,
        duel=True,
        seed=seed
    )

    scores = []
    moving_scores = []
    scores_window = deque(maxlen=100)
    start_time = time.time()
    total_steps = 0

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for _ in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            total_steps += 1
            state = next_state
            score += reward
            if done:
                break

        eps = max(eps * eps_decay, end_eps)

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

    plot_scores(env_name, [scores, moving_scores], 'result/dqn_log.png')
    agent.save_network('result/network.pth')


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
