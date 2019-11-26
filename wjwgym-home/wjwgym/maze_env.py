#!/usr/bin/env python
# coding=utf-8
"""
@create time: 2019-11-22 15:01
@author: Jiawei Wu
@edit time: 2019-11-25 21:28
@file: /maze.py
"""

import numpy as np
from gym.spaces import Discrete, Box
import gym


class MazeEnv(gym.Env):
    def __init__(self, height=4, width=4, hells=[(1, 2)], ovals=[(2, 1)]):
        """
        初始化迷宫环境，长宽都是5
        """
        self.road, self.player, self.hell, self.oval = 0, 1, 2, 3   # 定义状态
        self.action_space = Discrete(4)
        self.observation_space = Box(low=-1, high=1, shape=(height, width))
        init_space = np.zeros(height, width)
        self.player_pos = (0, 0)
        self.hells, self.ovals = hells, ovals
        for h in hells:
            x, y = h
            init_space[x][y] = self.hell
        for o in ovals:
            x, y = o
            init_space[x][y] = self.oval
        self.init_space = init_space    # 初始化状态
        self.step_count = 0     # 初始化步数记录

    def reset(self):
        """重置迷宫环境"""
        self.player_pos = (0, 0)
        self.step_count = 0

    def get_obs(self):
        """获取当前状态"""
        obs = self.init_space.copy()
        x, y = self.player_pos
        obs[x][y] = self.player
        return obs

    def get_reward(self):
        """获取奖励"""
        death, move, win = -1, 0, 1  # 定义不同状态的奖励
        if any(self.player_pos == h for h in self.hells):
            return death
        if any(self.player_pos == o for o in self.ovals):
            return win
        return move

    def get_done(self):
        """获取是否结束"""
        return any(self.player_pos == h for h in self.hells) or any(self.player_pos == o for o in self.ovals)

    def get_info(self):
        """获取额外信息"""
        return self.step_count

    def step(self, action):
        """
        执行一步动作，在迷宫环境中是向给定的方向走一步
        我们规定[up down left right] 分别为 [0 1 2 3]
        @param: action 传入的动作
        """
        action = np.clip(int(action), 0, 3)  # 限制action是整数且范围不能超过range(4)
        actions = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)
        }   # 定义action对应的x和y值变化
        next_pos = self.player_pos + actions[action]
        if all(self.observation_space.low < pos < self.observation_space.high for pos in next_pos):
            # 只有x和y都在范围内才算这一步有效
            self.player_pos = next_pos

        obs, reward, done, info = self.get_obs(), self.get_reward(), self.get_done(), self.get_info()
        return obs, reward, done, info

    def render(self):
        pass
