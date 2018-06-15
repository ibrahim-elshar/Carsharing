# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 15:40:44 2018

@author: IJE8
"""

import gym
import gym.spaces
import gym_carsharing
import numpy as np
env = gym.make('Carsharing-v0')

env.reset()
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
print(observation, reward, done, info)

env.reset()
returns=0
for i in range(12):
    print(env.x)
    print(env.s)
    action = env.action_space.sample()
    print(action)
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    returns+=reward
    print(i)
    print(returns)
    if done: 
        env.reset()
