#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-11-17 11:23
@edit time: 2019-12-07 20:26
@file: /dqn.py
@desc: 创建DQN对象
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gym
from wjwgym.agents.Utils import ExpReplay
CUDA = torch.cuda.is_available()


class DQNBase(object):
    def __init__(self, n_states, n_actions, learning_rate=0.01, discount_rate=0.9):
        """
        初始化DQN的两个网络和经验回放池
        @param n_obs: number of observations
        @param n_actions: number of actions
        """
        # DQN 的超参
        self.gamma = discount_rate  # 未来折扣率

        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.eval_every = 10
        # 网络创建
        self.n_states, self.n_actions = n_states, n_actions
        self._build_net()

        self.replay_buff = ExpReplay(n_states, 1, MAX_MEM=200)
        # 定义优化器和损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        # 记录步数用于同步参数
        self.eval_step = 0
        if CUDA:
            self.cuda()

    def _build_net(self):
        raise TypeError("Network build no implementation")

    def get_action(self, state):
        # epsilon update
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon
        # 将行向量转为列向量（1 x n_states -> n_states x 1 x 1)
        if np.random.rand() < self.epsilon:
            # 概率随机
            return np.random.randint(self.n_actions)
        else:
            # greedy
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_values = self.eval_net.forward(state).cpu()
            return action_values.data.numpy().argmax()

    def get_raw_out(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_values = self.eval_net.forward(state)
        return action_values

    def add_step(self, cur_state, action, reward, done, next_state):
        # 变量拼接
        step = np.hstack([cur_state, action, reward, done, next_state])
        self.replay_buff.add_step(step)

    def learn(self):
        bench = self.replay_buff.get_bench_splited_tensor(CUDA)
        if bench is None:
            return
        # 参数复制
        if self.eval_step % self.eval_every == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # 更新训练步数
        self.eval_step += 1
        # 拆分bench
        bench_cur_states, bench_actions, bench_rewards, bench_dones, bench_next_states = bench
        # 计算误差
        q_eval = self.eval_net(bench_cur_states)
        q_eval = q_eval.gather(1, bench_actions.long())  # shape (batch, 1)
        q_next = self.target_net(bench_next_states).detach()     # detach from graph, don't backpropagate
        # 如果done，则不考虑未来
        q_target = bench_rewards + self.gamma * (1 - bench_dones) * q_next.max(1)[0].view(len(bench_next_states), 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # 网络更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def cuda(self):
        self.eval_net.cuda()
        self.target_net.cuda()
