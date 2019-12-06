#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-04 10:36
@edit time: 2019-12-06 15:55
@file: ./DDPG_torch.py
"""
import numpy as np
from Utils import ExpReplay, soft_update
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import gym
import time

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


class DDPG(object):
    def __init__(self, n_states, n_actions, a_bound=1, lr_a=0.001, lr_c=0.002, tau=0.01, gamma=0.9):
        # 参数复制
        self.n_states, self.n_actions = n_states, n_actions
        self.tau, self.gamma = tau, gamma
        # 初始化训练指示符
        self.start_train = False
        self.mem_size = 0
        # 创建经验回放池`
        self.memory = ExpReplay(n_states,  n_actions, MAX_MEM=10000)  # s, a, r, d, s_
        # 创建神经网络
        self.actor_eval = Anet(n_states, n_actions, a_bound)
        self.actor_target = Anet(n_states, n_actions, a_bound)
        self.critic_eval = Cnet(n_states, n_actions)
        self.critic_target = Cnet(n_states, n_actions)
        if CUDA:
            self.cuda()
        # 指定优化器和损失函数
        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=lr_a)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=lr_c)
        self.mse_loss = nn.MSELoss()
        self.store = np.zeros((10000, n_states * 2 + n_actions + 2), dtype=np.float32)

    def choose_action(self, s):
        """给定当前状态，获取选择的动作"""
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.actor_eval.forward(s).detach().cpu()
        # action = np.argmax(actions)
        # action = np.random.choice(np.array([0,1,2,3]), p=actions[0])
        return action[0]

    def learn(self):
        """训练网络"""
        # 将eval网络参数赋给target网络
        soft_update(self.actor_target, self.actor_eval, self.tau)
        soft_update(self.critic_target, self.critic_eval, self.tau)

        # 获取bench并拆解
        # bench = self.memory.get_bench_splited_tensor(32)
        bench = self.memory.get_bench_splited_tensor(CUDA, 32)
        if bench is None:
            return
        else:
            self.start_train = True
        bench_cur_states, bench_actions, bench_rewards, bench_dones, bench_next_states = bench

        # 计算target_q，指导cirtic更新
        # 通过a_target和next_state计算target网络会选择的下一动作 next_action；通过target_q和next_states、刚刚计算的next_actions计算下一状态的q_values
        target_q_next = self.critic_target(bench_next_states, self.actor_target(bench_next_states))
        target_q = bench_rewards + self.gamma * (1 - bench_dones) * target_q_next   # 如果done，则不考虑未来
        # 指导critic更新
        q_value = self.critic_eval(bench_cur_states, bench_actions)
        td_error = self.mse_loss(target_q, q_value)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()

        # 指导actor更新
        policy_loss = self.critic_eval(bench_cur_states, self.actor_eval(bench_cur_states))  # 用更新的eval网络评估这个动作
        # 如果 a是一个正确的行为的话，那么它的policy_loss应该更贴近0
        loss_a = -torch.mean(policy_loss)
        self.actor_optim.zero_grad()
        loss_a.backward()
        self.actor_optim.step()

    def add_step(self, s, a, r, d, s_):
        step = np.hstack((s.reshape(-1), a, [r], [d], s_.reshape(-1)))
        self.memory.add_step(step)
        self.mem_size += 1

    def cuda(self):
        self.actor_eval.cuda()
        self.actor_target.cuda()
        self.critic_eval.cuda()
        self.critic_target.cuda()


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
            # ddpg.store_transition(s, a, r / 10, done, s_)

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
