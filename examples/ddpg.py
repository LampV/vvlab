#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-06 23:01
@edit time: 2019-12-06 23:13
@file: /test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gym
from wjwgym import DDPGBase
CUDA = torch.cuda.is_available()


class Anet(nn.Module):
    """定义Actor的网络结构"""

    def __init__(self, n_states, n_actions, a_bound):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        @param a_bound: bound of actino
        """
        super(Anet, self).__init__()
        n_neurons = 32
        self.fc1 = nn.Linear(n_states, n_neurons)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(n_neurons, n_actions)
        self.out.weight.data.normal_(0, 0.1)
        if CUDA:
            self.bound = torch.FloatTensor(a_bound).cuda()
        else:
            self.bound = torch.FloatTensor(a_bound)

    def forward(self, x):
        """
        定义网络结构: 第一层网络->ReLU激活->输出层->tanh激活->softmax->输出
        """
        x = x.cuda() if CUDA else x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        action_value = F.tanh(x)
        action_value = action_value * self.bound
        return action_value


class Cnet(nn.Module):
    """定义Critic的网络结构"""

    def __init__(self, n_states, n_actions):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        """
        super(Cnet, self).__init__()
        n_neurons = 30
        self.fc_state = nn.Linear(n_states, n_neurons)
        self.fc_state.weight.data.normal_(0, 0.1)
        self.fc_action = nn.Linear(n_actions, n_neurons)
        self.fc_action.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(n_neurons, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        """
        定义网络结构: 
        state -> 全连接   -·-->  中间层 -> 全连接 -> ReLU -> Q值
        action -> 全连接  /相加，偏置
        """
        s, a = (s.cuda(), a.cuda()) if CUDA else (s, a)
        x_s = self.fc_state(s)
        x_a = self.fc_action(a)
        x = F.relu(x_s+x_a)
        actions_value = self.out(x)
        return actions_value


class DDPG(DDPGBase):
    def _build_net(self):
        n_states, n_actions = self.n_states, self.n_actions
        self.actor_eval = Anet(n_states, n_actions, self.bound)
        self.actor_target = Anet(n_states, n_actions, self.bound)
        self.critic_eval = Cnet(n_states, n_actions)
        self.critic_target = Cnet(n_states, n_actions)



def rl_loop():
    ENV_NAME = 'Pendulum-v0'
    RENDER = False
    MAX_EPISODES = 10000
    MAX_EP_STEPS = 200

    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(s_dim, a_dim, a_bound)
    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.add_step(s, a, r / 10, done, s_)

            if ddpg.mem_size > 10000:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300:
                    RENDER = True
                break

    print('Running time: ', time.time() - t1)


if __name__ == '__main__':
    rl_loop()
