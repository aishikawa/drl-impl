import gym
import numpy as np
from dqn_agent import DqnAgent
from collections import deque

import matplotlib.pyplot as plt


def main(seed=1):
    n_episodes = 2000
    max_t = 10000
    eps = 0.01

    env = gym.make('CartPole-v1')
    env.seed(seed)
    agent = DqnAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    scores = []
    moving_scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        moving_score = np.mean(scores_window)
        moving_scores.append(moving_score)

        end = '\n' if i_episode % 100 == 0 else ''
        print(f'\rEpisode {i_episode}\tAverage Score: {moving_score:.2f}', end=end)

    plot_scores([scores, moving_scores])


def plot_scores(scores_list):
    n = np.arange(len(scores_list[0]))
    for scores in scores_list:
        plt.plot(n, scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()
    plt.savefig('dqn2.png')


if __name__ == '__main__':
    main()