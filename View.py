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
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import CarEnvDistanceReward
import DeepQNetwork

if __name__ == "__main__":

    savepath = Path('model/model_DQN_EnvDistanceReward_300-1635331329.0202696.pch')
    with savepath.open('rb') as file:
        model = T.load(file)

    episode = 10
    render = True
    #env = CarEnv()
    env = CarEnvDistanceReward(reward_function=reward_function,EXPERIENCE_SECONDE= 10)
    start_time = time.time()
    for i in range(episode):
        score = 0
        done = False
        observation = env.reset()
        #put in a shape of 1 and normalize
        flat_observation = observation.reshape(1,-1)[0]/255.0
        try : 
            while not done:

                if render:
                    #Show Preview
                    cv2.imshow(f'Agent - preview', observation)
                    cv2.waitKey(1)

                data = T.tensor(flat_observation).float()

                action = model.forward(data)
                action = action.detach().numpy().argmax()

                observation_, reward, done, info = env.step(action)
                flat_observation_ = observation_.reshape(1,-1)[0]/255.0
                score += reward
                flat_observation = flat_observation_
                observation = observation_

        finally : 
            if render:
                cv2.destroyWindow(f'Agent - preview')
            for actor in env.actor_list:
                actor.destroy()
            time_n = time.time() - start_time
            print('episode ', i, 'score %.2f' % score,'time %.2f s' % time_n)