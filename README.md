# drl-impl
This is an implementation of deep reinforcement learning algorithms.


### Algorithms
- [x] DQN
    - [x] Double Q-learning
    - [x] Dueling Network
    - [x] Prioritised Replay
    - [x] Multi-step learning
- [x] DDPG

## Install requirements
```
$ pip install -r requirements.txt
```

## Usage
Run DQN for [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
```
$ python run.py --algo DQN
```

Run DDPG for LunarLanderContinuous-v3
```
$ python run.py --algo DDPG
```

## Result
#### DQN
![dqn_log](docs/img/dqn_log.png)
#### DDPG
![ddpg_log](docs/img/ddpg_log.png)