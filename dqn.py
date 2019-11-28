#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-11-17 11:23
@edit time: 2019-11-28 10:30
@file: /dqn.py
@desc: 创建DQN对象
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gym
import wjwgym
from functools import reduce
from tqdm import trange

class ExpReplay:
    def __init__(self, dim, MAX_MEM=2000, MIN_MEM=None, BENCH_SIZE=None):
        """
        定义经验回放池的参数
        @param dim: Dimension of step (state, action, reward, done, next_state)
        @param MAX_MEM: Maximum memory
        @param MIN_MEM: Minimum memory，超过这个数目才返回bench
        @param BENCH_SIZE: Benchmark size
        """
        # 保证参数被定义
        self.dim = dim
        if not MIN_MEM:
            MIN_MEM = MAX_MEM // 10
        if not BENCH_SIZE:
            BENCH_SIZE = MIN_MEM // 2
        self.max_mem = MAX_MEM
        self.min_mem = MIN_MEM
        self.bench_size = BENCH_SIZE
        # 定义经验回放池
        self.expreplay_pool = np.array([[]])
        self.mem_count = 0

    # TODO 修复add_step失败的问题
    def add_step(self, step):
        self.expreplay_pool = np.append(self.expreplay_pool, step).reshape(-1, self.dim)
        if self.expreplay_pool.size > self.max_mem:
            # 如果超了，随机删除10%
            del_indexs = np.random.choice(self.max_mem, self.max_mem // 10)
            np.delete(self.expreplay_pool, del_indexs, axis=1)

    def get_bench(self, BENCH_SIZE=None):
        bench_size = BENCH_SIZE if BENCH_SIZE else self.bench_size
        if self.expreplay_pool.shape[0] > self.min_mem:
            # 比最小输出阈值大的时候才返回bench
            choice_indexs = np.random.choice(self.expreplay_pool.shape[0], bench_size)
            return self.expreplay_pool[choice_indexs]
        else:
            return None


class Net(nn.Module):
    """定义DQN的网络结构"""

    def __init__(self, n_states, n_actions):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        """
        super(Net, self).__init__()
        n_neurons = 32
        self.fc1 = nn.Linear(n_states, n_neurons)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(n_neurons, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        """
        定义网络结构: 第一层网络->ReLU激活->输出层->softmax->输出
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        action_values = F.softmax(x)
        return action_values


class DQN(object):
    def __init__(self, n_states, n_actions, learning_rate=0.01, discount_rate=0.9):
        """
        初始化DQN的两个网络和经验回放池
        @param n_obs: number of observations
        @param n_actions: number of actions
        """
        # DQN 的超参
        self.gamma = discount_rate # 未来折扣率

        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        self.eval_every = 10
        # 网络创建
        self.n_states = n_states
        self.n_actions = n_actions
        self.eval_net = Net(n_states, n_actions)
        self.target_net = Net(n_states, n_actions)
        self.replay_buff = ExpReplay(n_states * 2 + 3, 200)
        # 定义优化器和损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        # 记录步数用于同步参数
        self.eval_step = 0

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
            action_values = self.eval_net.forward(state)
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
        bench = self.replay_buff.get_bench()
        if bench is None:
            return
        # 参数复制
        if self.eval_step % self.eval_every == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # 更新训练步数
        self.eval_step += 1
        # 拆分bench
        bench_cur_states = torch.FloatTensor(bench[:, :self.n_states])
        bench_actions = torch.LongTensor(bench[:, self.n_states: self.n_states + 1].astype(int))
        bench_rewards = torch.FloatTensor(bench[:, self.n_states + 1: self.n_states + 2])
        bench_dones = torch.FloatTensor(bench[:, self.n_states + 2: self.n_states + 3].astype(int)) # 将是否结束按int类型读取，结束则为1，否则为0
        bench_next_states = torch.FloatTensor(bench[:, -self.n_states:])
        # 计算误差
        q_eval = self.eval_net(bench_cur_states)
        q_eval = q_eval.gather(1, bench_actions)  # shape (batch, 1)
        q_next = self.target_net(bench_next_states).detach()     # detach from graph, don't backpropagate
        # 如果done，则不考虑未来
        q_target = bench_rewards + self.gamma * (1 - bench_dones) * q_next.max(1)[0].view(len(bench), 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # 网络更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

def rl_loop():
    MAX_EPISODES  = 80
    env = gym.make('Maze-v0')
    n_states = reduce(np.multiply, env.observation_space.shape)
    n_actions = env.action_space.n
    agent = DQN(n_states, n_actions)
    # train
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((n_states))
        done = False
        while not done:
            action = agent.get_action(cur_state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape((n_states))
            agent.add_step(cur_state, action, reward, done, next_state)
            agent.learn()
            cur_state = next_state
        print('ep: ', ep, ' steps: ', info, ' final reward: ', reward)
    print('done')
    cur_state = env.reset()
    cur_state = cur_state.reshape((n_states))
    done = False
    # test 
    while not done:
        action = agent.get_action(cur_state)
        action_values = agent.get_raw_out(cur_state)
        print(action_values)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape((n_states))
        agent.add_step(cur_state, action, reward, done, next_state)
        agent.learn()
        cur_state = next_state

if __name__ == '__main__':
    rl_loop()
    