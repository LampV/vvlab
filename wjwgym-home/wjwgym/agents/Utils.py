#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-04 10:40
@edit time: 2019-12-16 17:10
@file: /exp_replay.py
"""

import torch
from torch.autograd import Variable
import numpy as np


class ExpReplay:
    def __init__(self, n_states, n_actions, MAX_MEM=2000, MIN_MEM=None, BATCH_SIZE=None):
        """
        定义经验回放池的参数
        @param dim: Dimension of step (state, action, reward, done, next_state)
        @param MAX_MEM: Maximum memory
        @param MIN_MEM: Minimum memory，超过这个数目才返回batch
        @param BATCH_SIZE: Batchmark size
        """
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
        """为经验回放池增加一步，一步通常包括s, a, r, d, s_"""
        self.expreplay_pool = np.append(self.expreplay_pool, step).reshape(-1, self.dim)
        if self.expreplay_pool.shape[0] > self.max_mem:
            # 如果超了，随机删除10%
            del_indexs = np.random.choice(self.max_mem, self.max_mem // 10)
            np.delete(self.expreplay_pool, del_indexs, axis=0)

    def get_batch(self, BATCH_SIZE=None):
        """
        从回放池中获取一个batch
        如果batch_size未指定则按创建时的大小返回；
        如果经验回放池大小未达到上限则返回None
        """
        batch_size = BATCH_SIZE if BATCH_SIZE else self.batch_size
        if self.expreplay_pool.shape[0] > self.min_mem:
            # 比最小输出阈值大的时候才返回batch
            choice_indexs = np.random.choice(self.expreplay_pool.shape[0], batch_size)
            return self.expreplay_pool[choice_indexs]
        else:
            return None

    def get_batch_splited(self, BATCH_SIZE=None):
        """将batch按照s, a, r, d, s_的顺序分割好并返回"""
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
        """将batch分割并转换为tensor之后返回"""
        batch = self.get_batch_splited(BATCH_SIZE)
        if batch is None:
            return batch
        else:
            return (Variable(torch.from_numpy(ndarray)).type(dtype).cuda() for ndarray in batch) if CUDA else (Variable(torch.from_numpy(ndarray)).type(dtype) for ndarray in batch)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )