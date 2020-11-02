#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-11-17 11:23
@edit time: 2020-05-10 22:17
@FilePath: /vvlab/agents/DQN_base.py
@desc: 创建DQN对象
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gym
from ..utils import ReplayBuffer
CUDA = torch.cuda.is_available()


class DQNBase(object):
    def __init__(self, n_states, n_actions, learning_rate=0.001, discount_rate=0.0, card_no=0, **kwargs):
        """
        初始化DQN的两个网络和经验回放池
        @param n_obs: number of observations
        @param n_actions: number of actions
        """
        # DQN 的超参
        self.gamma = discount_rate  # 未来折扣率

        self.epsilon = 0.6
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.eval_every = 10
        self.card_no = card_no
        # 网络创建
        self.n_states, self.n_actions = n_states, n_actions
        self._build_net()

         self.buff_size, self.buff_thres, self.batch_size = 50000, 1000, 256
        if 'buff_size' in kwargs:
            self.buff_size = kwargs['buff_size']
        if 'buff_thres' in kwargs:
            self.buff_thres = kwargs['buff_thres']
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        self.replay_buff = ReplayBuffer(n_states, 1, buff_size=self.buff_size,
                                        buff_thres=self.buff_thres, batch_size=self.batch_size, card_no=self.card_no)
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
            action_size = state.shape[0]
            return np.random.randint(0, self.n_actions, (1, action_size))
        else:
            # greedy
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_values = self.eval_net.forward(state).cpu()
            return action_values.data.numpy().argmax(axis=len(action_values.shape)-1)

    def get_raw_out(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_values = self.eval_net.forward(state)
        return action_values

    def add_step(self, cur_state, action, reward, done, next_state):
        self.replay_buff.add_step(cur_state, action, reward, done, next_state)

    def learn(self):
        batch = self.replay_buff.get_batch_splited_tensor(CUDA)
        if batch is None:
            return
        # 参数复制
        if self.eval_step % self.eval_every == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # 更新训练步数
        self.eval_step += 1
        # 拆分batch
        batch_cur_states, batch_actions, batch_rewards, batch_dones, batch_next_states = batch
        # 计算误差
        q_eval = self.eval_net(batch_cur_states)
        q_eval = q_eval.gather(1, batch_actions.long())  # shape (batch, 1)
        q_next = self.target_net(batch_next_states).detach()     # detach from graph, don't backpropagate
        # 如果done，则不考虑未来
        q_target = batch_rewards + self.gamma * (1 - batch_dones) * q_next.max(1)[0].view(len(batch_next_states), 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # 网络更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def cuda(self):
        self.eval_net.cuda(self.card_no)
        self.target_net.cuda(self.card_no)
