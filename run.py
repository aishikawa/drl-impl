import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse

from dqn.dqn_agent import DqnAgent
from ddpg.ddpg_agent import DdpgAgent


def make_env_and_agent(algo, seed):
    if algo == 'DQN':
        env_name = 'LunarLander-v2'
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DqnAgent(state_dim, action_size, gamma=0.99, soft_target_update=True, double=True, duel=True, seed=seed)
    elif algo == 'DDPG':
        env_name = 'LunarLanderContinuous-v2'
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DdpgAgent(state_dim, action_dim, gamma=0.99, random_seed=seed)

    env.seed(0)
    env = gym.wrappers.Monitor(env, f'result/{algo}/video/', force=True)

    return env, env_name, agent


def plot_scores(env_name, scores_list, filename):
    n = np.arange(len(scores_list[0]))
    for scores in scores_list:
        plt.plot(n, scores)
    plt.title(env_name)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(filename)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['DQN', 'DDPG'], default='DDPG')

    args = parser.parse_args()
    env, env_name, agent = make_env_and_agent(args.algo, seed=1)

    n_episodes = 1001
    max_t = 1000

    scores = []
    moving_scores = []
    scores_window = deque(maxlen=100)
    start_time = time.time()
    total_steps = 0

    for i_episode in range(1, n_episodes+1):
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

        agent.end_episode()

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

    plot_scores(env_name, [scores, moving_scores], f'result/{args.algo}/log.png')
    agent.save_network(f'result/{args.algo}')


if __name__ == '__main__':
    main()
