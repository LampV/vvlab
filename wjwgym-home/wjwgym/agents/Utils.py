#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-04 10:40
@edit time: 2020-01-14 11:40
@file: /exp_replay.py
"""

import torch
from torch.autograd import Variable
import numpy as np


class ExpReplay:
    """
    使用类似Q-Learning算法的DRL通用的经验回放池
    1. 经验回放池的一条记录
        (state, action, reward, done, next_state) 或简写为 (s, a, r, d, s_)
        其中done的意义在于，当状态为done的时候，Q估计=r，而不是Q估计=(r + gamma * Q'max)
        显然，一条记录的维度是 dim(s) + dim(a) + dim(r) + dim(d) + dim(s_) = 2 * n_states + n_actions + 2
    2. 经验回放池的参数
        - n_states: state的维度
        - n_actions: action的维度
        - MAX_MEM: 经验回放池的容量上限。达到这个上限之后，后续的记录就会覆盖之前的记录。
            默认值是: 2000
            TODO 将默认值改为1000
        - MIN_MEN: 经验回放池输出的最小数目。记录数目超过阈值之后，获取batch才会获得输出
            默认值是: MAX_MEM//10
        - BATCH_SIZE: 一个batch的大小。
            默认值是: MAX_MEM//20
            TODO 将默认值改为32
    3. 经验回放池的功能
        - 创建经验回放池对象
            >>> self.memory = ExpReplay(n_states,  n_actions, MAX_MEM=MAX_MEM, MIN_MEM=MIN_MEM)
        - 添加一条记录
            >>> step = np.hstack((s.reshape(-1), a, [r], [d], s_.reshape(-1)))
            >>> self.memory.add_step(step)
            TODO 将hstack放在ExpReplay.add_step内部进行，对外提供add_step(s, a, r, d, s_)接口
            每条记录都会被添加到经验回放池
            如果添加之后超过了回放池的上限，则会随机删除 10%
            TODO 将这个规则改为通用的顺序覆盖规则？
        - 获取一个batch用于训练
            >>> s, a, r, d, s_ = memery.get_batch_splited()
        - 获取一个已经被转为pytorch Variable的batch用于训练
            >>> CUDA = torch.cuda.is_available()
            >>> batch = self.memory.get_batch_splited_tensor(CUDA, batch_size)
    """

    def __init__(self, n_states, n_actions, MAX_MEM=2000, MIN_MEM=None, BATCH_SIZE=None):
        """初始化经验回放池"""
        # 保证参数被定义
        self.n_states, self.n_actions = n_states, n_actions
        self.dim = n_states * 2 + n_actions + 2
        if not MIN_MEM:
            MIN_MEM = MAX_MEM // 10
        if not BATCH_SIZE:
            BATCH_SIZE = MIN_MEM // 2
        self.max_mem = MAX_MEM
        self.min_mem = MIN_MEM
        self.batch_size = BATCH_SIZE
        # 定义经验回放池
        self.expreplay_pool = np.array([[]])
        self.mem_count = 0

    def add_step(self, step):
        """
        为经验回放池增加一步，我们约定“一步”包括s, a, r, d, s_五个部分
        @param step: 一条需要被添加到经验回放池的记录
        """
        self.expreplay_pool = np.append(self.expreplay_pool, step).reshape(-1, self.dim)
        if self.expreplay_pool.shape[0] > self.max_mem:
            # 如果超了，随机删除10%
            del_indexs = np.random.choice(self.max_mem, self.max_mem // 10)
            np.delete(self.expreplay_pool, del_indexs, axis=0)

    def get_batch(self, BATCH_SIZE=None):
        """
        从回放池中获取一个batch
        如果经验回放池大小未达到输出阈值则返回None
        
        @param BATCH_SIZE: 一个batch的大小，若不指定则按经验回放池的默认值
        @return 一个batch size 的记录
        """
        batch_size = BATCH_SIZE if BATCH_SIZE else self.batch_size
        if self.expreplay_pool.shape[0] > self.min_mem:
            # 比最小输出阈值大的时候才返回batch
            choice_indexs = np.random.choice(self.expreplay_pool.shape[0], batch_size)
            return self.expreplay_pool[choice_indexs]
        else:
            return None

    def get_batch_splited(self, BATCH_SIZE=None):
        """
        将batch按照s, a, r, d, s_的顺序分割好并返回
        
        @param BATCH_SIZE: 一个batch的大小，若不指定则按经验回放池的默认值
        @return cur_states, actions, rewards, dones, nexe_states: 
            按照 (s, a, r, d, s_) 顺序分割好的一组batch 
        """
        batch = self.get_batch(BATCH_SIZE)
        if batch is None:
            return batch
        else:
            cur_states = batch[:, :self.n_states]
            actions = batch[:, self.n_states: self.n_states + self.n_actions].astype(int)
            rewards = batch[:, self.n_states + self.n_actions: self.n_states + self.n_actions + 1]
            dones = batch[:, -self.n_states - 1: -self.n_states].astype(int)  # 将是否结束按int类型读取，结束则为1，否则为0
            nexe_states = batch[:, -self.n_states:]
            return cur_states, actions, rewards, dones, nexe_states

    def get_batch_splited_tensor(self, CUDA, BATCH_SIZE=None, dtype=torch.FloatTensor):
        """
        将batch分割并转换为tensor之后返回
        
        @param CUDA: 是否使用GPU，这决定了返回变量的设备类型
        @param BATCH_SIZE: 一个batch的大小，若不指定则按经验回放池的默认值
        @param dtype: 返回变量的数据类型，默认为Float
        @return cur_states, actions, rewards, dones, nexe_states: 
            按照 (s, a, r, d, s_) 顺序分割好且已经转为torch Variable的一组batch 
        """
        batch = self.get_batch_splited(BATCH_SIZE)
        if batch is None:
            return batch
        else:
            return (Variable(torch.from_numpy(ndarray)).type(dtype).cuda() for ndarray in batch) if CUDA else (Variable(torch.from_numpy(ndarray)).type(dtype) for ndarray in batch)


class OUProcess(object):
    """Ornstein-Uhlenbeck process"""

    def __init__(self, x_size, mu=0, theta=0.15, sigma=0.3):
        self.x = np.ones(x_size) * mu
        self.x_size = x_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def noise(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.x_size)
        self.x = self.x + dx
        return self.x


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
