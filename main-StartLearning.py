PATH_CARLA_EGG = '../DRL/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg'

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import gym

from gym import Env
from gym.spaces import Discrete, Box 

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

try:
    sys.path.append(glob.glob(PATH_CARLA_EGG % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

if __name__ == "__main__":
    env = CarEnvDistanceReward()
    #env = CarEnvDistanceReward(reward_function=reward_function)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=5, n_actions=3, eps_end=0.1,
                  input_dims=[480*640*3], lr=0.001, eps_dec=1e-3)
    scores, eps_history = [], []
    n_games = 300
    render = False

    start_time = time.time()
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        flat_observation = observation.reshape(1,-1)[0]/255.0
        while not done:
            action = agent.choose_action(flat_observation.astype(float))
            if render:
                cv2.imshow(f'Agent - preview', observation)
                cv2.waitKey(1)
            
            observation_, reward, done, info = env.step(action)
            flat_observation_ = observation_.reshape(1,-1)[0]/255.0
            score += reward
            agent.store_transition(flat_observation.astype(float), action, reward, 
                                    flat_observation_, done)
            agent.learn()
            flat_observation = flat_observation_
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-10:])
        if render:
            cv2.destroyWindow(f'Agent - preview')
        for actor in env.actor_list:
            actor.destroy()
        time_n = time.time() - start_time
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon,
                 'time %.2f s' % time_n)

    savepath = Path('model/model_DQN_EnvDistanceReward_WithRewardFunction_{}.pch'.format(str(n_games)+'-'+str(start_time)))


    with savepath.open('wb') as file:
        T.save(agent.Q_eval, file)