#!/usr/bin/env python
# coding=utf-8
"""
@create time: 2019-11-22 10:19
@author: Jiawei Wu
@edit time: 2019-11-22 14:55
@file: /linear_agent.py
"""

import numpy as np
import pandas as pd


class LinearAgent:
    """线性学习（QLearning、Sarsa及变形）的基类"""
    def __init__(self, actions: list, e_greedy: float < 1, learning_rate, reward_decay):
        """
        初始化基类
        @param actions: 动作空间（注：规定动作都是非负整数）
        @param e_greedy: 选取最大价值动作的概率epsilon-greedy初始值
        @param learning_rate: 学习率
        @param reward_decay: 未来折扣率
        """
        self.actions = actions  # a list
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.lr = learning_rate
        self.gamma = reward_decay

    def choose_action(self, observation, epsilon=None):
        """
        获取当前状态对应的动作。按照epsilon的概率从Q表读取价值最大的，剩下情况随机选取。
        @param observation: 当前状态
        @param epsilon: epsilon值，如果没有传递这个值，则按照开始设置的默认值随机。
        """
        # 确保epsilon被设置
        if not epsilon:
            epsilon = self.epsilon
        # 检测当前状态是否存在（若不存在会被初始化）
        self.check_state_exist(observation)
        
        # 选取动作
        if np.random.uniform() < epsilon:
            # 获取当前state对应的QTable行
            state_actions = self.q_table.loc[observation, :]
            # 在所有value最大的index中随机选取
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        else:
            # 随机选择
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        """
        检测状态是否存在，如果不存在则添加到QTable的行，并初始化为全0
            :param self: 
            :param state: 
        """   
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def print_table(self):
        """打印QTable内容"""
        q_table = self.q_table
        print(q_table)


class QLearning(LinearAgent):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearning, self).__init__(actions, e_greedy, learning_rate, reward_decay)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


class Sarsa(LinearAgent):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearning, self).__init__(actions, e_greedy)
        self.lr = learning_rate
        self.gamma = reward_decay

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    
