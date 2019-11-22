#!/usr/bin/env python
# coding=utf-8
"""
@create time: 2019-11-22 15:01
@author: Jiawei Wu
@edit time: 2019-11-22 15:04
@file: /maze.py
"""


import numpy as np
from space import Discreate

class maze_env:
    def __init__(self, height=5, width=5, hells=[], ovals=[]):
        self.player_pos = 0

        player, road, treasure = -1, 0, 1
        obs_states = [player, road, treasure]
        self.player, self.road, self.treasure = obs_states
        obs = Discreate(length, min(obs_states), max(obs_states))
        init_obs = np.array([self.road] * length)
        self.init_obs = init_obs.copy()
        init_obs[self.player_pos] = self.player
        init_obs[length - 1] = self.treasure
        obs.set_data(init_obs)
        self.observation_space = obs

        head_left, head_right = 0, 1
        ac_states = [head_left, head_right]
        ac_space = Discreate(1, min(ac_states), max(ac_states))
        self.action_space = ac_space

        self.step_count = 0
    
    def reset(self):
        self.player_pos = 0
        init_obs = self.init_obs.copy()
        init_obs[self.player_pos] = self.player
        init_obs[self.observation_space.shape - 1] = self.treasure
        self.observation_space.set_data(init_obs)
        self.step_count = 0

        return self.observation_space.data

    def step(self, action):
        if action == 0:
            delta = -1
        else:
            delta = 1
        next_pos = self.player_pos + delta
        if 0 <= next_pos < self.observation_space.shape:
            self.player_pos = next_pos
 
        obs = self.init_obs.copy()
        obs[self.observation_space.shape - 1] = self.treasure
        obs[self.player_pos] = self.player

        self.observation_space.set_data(obs)
        reward = int(next_pos == self.observation_space.shape - 1)
        done = next_pos == self.observation_space.shape - 1
        self.step_count += 1
        info = self.step_count
        return self.observation_space.data, reward, done, info

    def render(self):
        ascii_map = {
            -1: '*',
            0: '-',
            1: 'o'
        }
        print(f'{"".join([ascii_map[d] for d in self.observation_space.data])}\r', end='')







    