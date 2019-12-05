#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-04 10:40
@edit time: 2019-12-05 11:00
@file: /exp_replay.py
"""


import numpy as np
class ExpReplay:
    def __init__(self, n_states, n_actions, MAX_MEM=2000, MIN_MEM=None, BENCH_SIZE=None):
        """
        定义经验回放池的参数
        @param dim: Dimension of step (state, action, reward, done, next_state)
        @param MAX_MEM: Maximum memory
        @param MIN_MEM: Minimum memory，超过这个数目才返回bench
        @param BENCH_SIZE: Benchmark size
        """
        # 保证参数被定义
        self.n_states, self.n_actions = n_states, n_actions
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

    def add_step(self, step):
        """为经验回放池增加一步，一步通常包括s, a, r, d, s_"""
        self.expreplay_pool = np.append(self.expreplay_pool, step).reshape(-1, )
        if self.expreplay_pool.size > self.max_mem:
            # 如果超了，随机删除10%
            del_indexs = np.random.choice(self.max_mem, self.max_mem // 10)
            np.delete(self.expreplay_pool, del_indexs, axis=1)

    def get_bench(self, BENCH_SIZE=None):
        """
        从回放池中获取一个bench
        如果bench_size未指定则按创建时的大小返回；
        如果经验回放池大小未达到上限则返回None
        """
        bench_size = BENCH_SIZE if BENCH_SIZE else self.bench_size
        if self.expreplay_pool.shape[0] > self.min_mem:
            # 比最小输出阈值大的时候才返回bench
            choice_indexs = np.random.choice(self.expreplay_pool.shape[0], bench_size)
            return self.expreplay_pool[choice_indexs]
        else:
            return None

    def get_bench_splited(self, BENCH_SIZE=None):
        """将bench按照s, a, r, d, s_的顺序分割好并返回"""
        bench = self.get_bench(BENCH_SIZE)
        if bench is None:
            return bench
        else:
            cur_states = bench[:, :self.n_states]
            actions = bench[:, self.n_states: self.n_states + self.n_actions].astype(int)
            rewards = bench[:, self.n_states + self.n_actions: self.n_states + self.n_actions + 1]
            dones = bench[:, self.n_states + self.n_actions + 1: self.n_states + self.n_actions + 2].astype(int) # 将是否结束按int类型读取，结束则为1，否则为0
            states = bench[:, -self.n_states:]
            return cur_states, actions, rewards, dones, states