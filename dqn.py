import gym
import numpy as np
from dqn_agent import DqnAgent
from collections import deque

import matplotlib.pyplot as plt
import time


def main(seed=1):
    n_episodes = 2000
    max_t = 10000
    eps = 1
    end_eps = 0.01
    eps_decay = 0.995

    env = gym.make('CartPole-v1')
    env.seed(seed)
    agent = DqnAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        soft_target_update=True,
        double=True,
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
        end = '\n' if i_episode % 100 == 0 else ''
        print(f'\rEpisode {i_episode}\tAverage Score: {moving_score:.2f}\tSteps/sec: {steps_per_sec:.2f}', end=end)

    plot_scores([scores, moving_scores], 'dqn_log.png')


def plot_scores(scores_list, filename):
    n = np.arange(len(scores_list[0]))
    for scores in scores_list:
        plt.plot(n, scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    main(0)