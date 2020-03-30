#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-04 10:36
@edit time: 2020-01-14 17:24
@file: ./DDPG_torch.py
"""
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from .Utils import ExpReplay, soft_update, OUProcess
CUDA = torch.cuda.is_available()


class DDPGBase(object):
    def __init__(self, n_states, n_actions, bound=1, exp_size=10000, exp_thres=None, batch_size=32,
                 lr_a=0.001, lr_c=0.002, tau=0.01, gamma=0.9,
                 summary=False, **kwargs):
        # 参数复制
        self.n_states, self.n_actions, self.bound = n_states, n_actions, bound
        self.exp_size, self.exp_thres, self.batch_size = exp_size, exp_thres, batch_size
        self.lr_a, self.lr_c, self.tau, self.gamma = lr_a, lr_c, tau, gamma
        self.summary = summary
        self.kwargs = kwargs
        
        # 初始化episode和step
        self.episode, self.step = 0, 0
        # 参数覆盖
        self._param_override()
        
        # 创建经验回放池
        self.memory = ExpReplay(self.n_states, self.n_actions, exp_size=self.exp_size, exp_thres=self.exp_thres)
        
        # 创建神经网络
        self._build_net()
        # 指定优化器
        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=self.lr_c)
        # 约定损失函数
        self.mse_loss = nn.MSELoss()
        
        # 指定噪声发生器
        self._build_noise()
        
        # 指定summary writer
        self._build_summary_writer()
        
        # 开启cuda
        if CUDA:
            self.cuda()
            
    def _param_override(self):
        """覆盖参数
        提供子类简单覆写基类参数的方法
        例如：修改summary是否开启
        应当谨慎使用这个方法
        """
        pass

    def _build_net(self):
        raise TypeError("网络构建函数未被实现")

    def _build_noise(self, *args):
        raise TypeError("噪声发生器构建函数未被实现")

    def _build_summary_writer(self):
        """构建summary_writer
        如果指定了不需要summary_writer，会将其置为None
        如果指定了保存路径就使用保存路径，否则使用默认路径
        """
        if self.summary:
            if 'summary_path' in self.kwargs:
                self.summary_writer = SummaryWriter(log_dir=kwargs['summary_path'])
                self._build_summary_writer(kwargs['summary_path'])
            else:
                self.summary_writer = SummaryWriter()
        else:
            self.summary_writer = None

    def get_summary_writer(self):
        return self.summary_writer

    def _get_action(self, s):
        """给定当前状态，获取选择的动作"""
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.actor_eval.forward(s).detach().cpu().numpy()
        return action

    def get_action(self, s):
        return self._get_action(s)

    def _save(self, save_path, append_dict={}):
        """保存当前模型的网络参数
        @param save_path: 模型的保存位置
        @param append_dict: 除了网络模型之外需要保存的内容
        """
        states = {
            'actor_eval_net': self.actor_eval.state_dict(),
            'actor_target_net': self.actor_target.state_dict(),
            'critic_eval_net': self.critic_eval.state_dict(),
            'critic_target_net': self.critic_target.state_dict(),
        }
        states.update(append_dict)
        torch.save(states, save_path)

    def save(self, episode=None, save_path='./cur_model.pth'):
        """保存的默认实现
        @param episode: 当前的episode
        @param save_path: 模型的保存位置，默认是'./cur_model.pth'
        """
        append_dict = {
            'episode': self.episode if episode is None else episode,
            'step': self.step
        }
        self._save(save_path, append_dict)

    def _load(self, save_path):
        """加载模型参数
        @param save_path: 模型的保存位置
        @return: 加载得到的模型字典
        """
        if CUDA:
            states = torch.load(save_path, map_location=torch.device('cuda'))
        else:
            states = torch.load(save_path, map_location=torch.device('cpu'))

        # 从模型中加载网络参数
        self.actor_eval.load_state_dict(states['actor_eval_net'])
        self.actor_target.load_state_dict(states['actor_target_net'])
        self.critic_eval.load_state_dict(states['critic_eval_net'])
        self.critic_target.load_state_dict(states['critic_target_net'])

        # 从模型中加载episode和step信息
        self.episode, self.step = states['episode'], states['step']
        # 返回states
        return states

    def load(self, save_path='./cur_model.pth'):
        """加载模型的默认实现
        @param save_path: 模型的保存位置, 默认是 './cur_model.pth'
        @return: 被记录的episode值
        """
        print('\033[1;31;40m{}\033[0m'.format('加载模型参数...'))
        if not os.path.exists(save_path):
            print('\033[1;31;40m{}\033[0m'.format('没找到保存文件'))
            return -1
        else:
            states = self._load(save_path)
            return states['episode']

    def learn(self):
        """训练网络"""
        # 将eval网络参数赋给target网络
        soft_update(self.actor_target, self.actor_eval, self.tau)
        soft_update(self.critic_target, self.critic_eval, self.tau)

        # 获取batch并拆解
        batch = self.memory.get_batch_splited_tensor(CUDA, self.batch_size)
        if batch is None:
            return None, None
        else:
            self.start_train = True
        batch_cur_states, batch_actions, batch_rewards, batch_dones, batch_next_states = batch
        # 计算target_q，指导cirtic更新
        # 通过a_target和next_state计算target网络会选择的下一动作 next_action；通过target_q和next_states、刚刚计算的next_actions计算下一状态的q_values
        target_q_next = self.critic_target(batch_next_states, self.actor_target(batch_next_states))
        target_q = batch_rewards + self.gamma * (1 - batch_dones) * target_q_next   # 如果done，则不考虑未来
        # 指导critic更新
        q_value = self.critic_eval(batch_cur_states, batch_actions)
        td_error = self.mse_loss(target_q, q_value)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()

        # 指导actor更新
        policy_loss = self.critic_eval(batch_cur_states, self.actor_eval(batch_cur_states))  # 用更新的eval网络评估这个动作
        # 如果 a是一个正确的行为的话，那么它的policy_loss应该更贴近0
        loss_a = -torch.mean(policy_loss)
        self.actor_optim.zero_grad()
        loss_a.backward()
        self.actor_optim.step()
        return td_error.detach().cpu().numpy(), loss_a.detach().cpu().numpy()

    def _add_step(self, s, a, r, d, s_):
        """向经验回放池添加一条记录"""
        self.memory.add_step(s, a, r, d, s_)
        
    def add_step(self, s, a, r, d, s_):
        """添加记录的默认实现
        除了添加记录之外不做任何操作
        """
        self._add_step(s, a, r, d, s_)

    def cuda(self):
        self.actor_eval.cuda()
        self.actor_target.cuda()
        self.critic_eval.cuda()
        self.critic_target.cuda()
