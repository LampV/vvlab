#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-04 10:36
@edit time: 2019-12-07 21:49
@file: ./DDPG_torch.py
"""
import numpy as np
from wjwgym.agents.Utils import ExpReplay, soft_update
import torch.nn as nn
import torch
CUDA = torch.cuda.is_available()


class DDPGBase(object):
    def __init__(self, n_states, n_actions, a_bound=1, lr_a=0.001, lr_c=0.002, tau=0.01, gamma=0.9, 
        MAX_MEM=10000, MIN_MEM=None, BENCH_SIZE=32):
        # 参数复制
        self.n_states, self.n_actions = n_states, n_actions
        self.tau, self.gamma, self.bound = tau, gamma, a_bound
        self.bench_size = BENCH_SIZE
        # 初始化训练指示符
        self.start_train = False
        self.mem_size = 0
        # 创建经验回放池
        self.memory = ExpReplay(n_states,  n_actions, MAX_MEM=MAX_MEM, MIN_MEM=MIN_MEM)  # s, a, r, d, s_
        # 创建神经网络并指定优化器
        self._build_net()
        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=lr_a)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=lr_c)
        # 约定损失函数
        self.mse_loss = nn.MSELoss()
        # 开启cuda
        if CUDA:
            self.cuda()

    def _build_net(self):
        raise TypeError("Network not Implemented")

    def choose_action(self, s):
        """给定当前状态，获取选择的动作"""
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.actor_eval.forward(s).detach().cpu()
        return action[0]

    def learn(self):
        """训练网络"""
        # 将eval网络参数赋给target网络
        soft_update(self.actor_target, self.actor_eval, self.tau)
        soft_update(self.critic_target, self.critic_eval, self.tau)

        # 获取bench并拆解
        bench = self.memory.get_bench_splited_tensor(CUDA, self.bench_size)
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
