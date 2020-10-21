#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: ,: Jiawei Wu
@create time: 2020-09-25 11:20
@edit time: ,: 2020-10-21 16:38
@FilePath: ,: /vvlab/vvlab/envs/power_allocation/pa_env.py
@desc: 
Created on Sat Sep 15 11:24:43 2018
Q / gamma = 0
minimum transmit power: 5dBm/ maximum: 38dBm
bandwidth 10MHz
AWGN power -114dBm
path loss 120.9+37.6log10(d) (dB) d: transmitting distance (km)
using interferers' set and therefore reducing the computation complexity
multiple users / single BS
FP algorithm, WMMSE algorithm, maximum power, random power allocation schemes as comparisons
downlink
"""
import scipy.special
import scipy.io
import scipy
import numpy as np
from pprint import pprint
from collections import namedtuple
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
Node = namedtuple('Node', 'x y type')


memory_size = 50000
INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0001
learning_rate = 0.001
train_interval = 10
batch_size = 256


class PAEnv:
    @property
    def n_states(self):
        """return dim of state"""
        part_len = {
            'power': self.m_state + 1,
            'rate': self.m_state + 1,
            'fading': self.m_state
        }
        return sum(part_len[metric] for metric in self.metrics)

    @property
    def n_actions(self):
        """return num of actions"""
        return self.n_levels

    def init_observation_space(self, kwargs):
        valid_parts = ['power', 'rate', 'fading']
        # default using power
        sorter = kwargs['sorter'] if 'sorter' in kwargs else 'power'
        # default using all
        metrics = kwargs['metrics'] if 'metrics' in kwargs else valid_parts

        # check data
        if sorter not in valid_parts:
            msg = f'sorter should in power, rate and fading, but is {sorter}'
            raise ValueError(msg)
        if any(metric not in valid_parts for metric in metrics):
            msg = f'metrics should in power, rate and fading, but is {metrics}'
            raise ValueError(msg)

        # set to instance attr
        self.sorter, self.metrics = sorter, metrics

    def init_power_levels(self):
        min_power, max_power = self.min_power, self.max_power
        zero_power = 0
        def cal_p(l): return 1e-3 * np.power(10, l / 10)
        dbm_powers = np.linspace(min_power, max_power, num=self.n_levels-1)
        powers = [cal_p(l) for l in dbm_powers]
        powers = [zero_power] + powers
        self.power_levels = np.array(powers)

    def init_pos(self):
        """初始化用户和设备的位置

        随机基站的m个用户的位置；随机d2d设备的位置。其中：
        用户位置均位于基站半径R_bs以内，且在基站保护半径r_bs以外；
        传输设备均位于基站半径(R_bs - R_dev)以内
        接收设备均位于传输设备半径(r_dev)以内
        """
        r_bs, R_bs, r_dev, R_dev = self.r_bs, self.R_bs, self.r_dev, self.R_dev

        self.station = Node(0, 0, 'station')
        self.users = {}
        for i in range(self.m_usr):
            rho, phi = np.random.uniform(
                r_bs, R_bs), np.random.uniform(-np.pi, np.pi)
            x, y = rho * np.cos(phi), rho * np.sin(phi)
            user = Node(x, y, 'user')
            self.users[i] = user
        self.devices = {}
        for t in range(self.n_t):
            rho, phi = np.random.uniform(
                r_bs, R_bs - R_dev), np.random.uniform(-np.pi, np.pi)
            x, y = rho * np.cos(phi), rho * np.sin(phi)
            t_device = Node(x, y, 't_device')
            r_devices = {}
            for r in range(self.m_r):
                d_rho, d_phi = np.random.uniform(
                    r_dev, R_dev), np.random.uniform(-np.pi, np.pi)
                d_x, d_y = d_rho * np.cos(d_phi), d_rho * np.sin(d_phi)
                x, y = t_device.x + d_x, t_device.y + d_y
                r_device = Node(x, y, 'r_device')
                r_devices[r] = r_device
            self.devices[t] = {'t_device': t_device, 'r_devices': r_devices}

    def init_jakes(self, fd=10, Ts=20e-3, Ns=50):
        """初始化Jakes模型

        Jakes模型是用来仿真瑞利信道的方式，使用正弦波叠加法。
        瑞利信道是快衰落模型，与距离无关（但是受到发送方和接收方位置影响）。
        因此特定send到特定recv可能有多个信号，但是每个信号的衰落是一致的。
        因为不考虑干扰阈值距离，所以衰落矩阵是一个n_recvs * n_recvs的矩阵。

        Args:
            fd: 多普勒频移, default 10
            Ts:= 采样间隔, default 20e-3
            Ns: 正弦波叠加数目, default 50
        """
        # d2d发送端/每个发送端的接收端/基站数/每个基站的用户数
        n_t, m_r, n_bs, m_usr = self.n_t, self.m_r, 1, self.m_usr
        n_recvs = n_t * m_r + n_bs * m_usr

        def calc_h_set(pho):
            # 使用Jakes模型计算其中一个正弦波的衰落
            h_d2d = np.kron(np.sqrt((1.-pho**2)*0.5*(np.random.randn(n_recvs, n_t) **
                                                     2+np.random.randn(n_recvs, n_t)**2)),
                            np.ones((1, m_r), dtype=np.int32))
            h_bs = np.kron(np.sqrt((1.-pho**2)*0.5*(np.random.randn(n_recvs, n_bs)**2 +
                                                    np.random.randn(n_recvs, n_bs)**2)),
                           np.ones((1, m_usr), dtype=np.int32))
            h_set = np.concatenate((h_d2d, h_bs), axis=1)
            return h_set
        # 迭代计算所有Ns组正弦波
        H_set = np.zeros([n_recvs, n_recvs, int(Ns)], dtype=np.float32)
        pho = np.float32(scipy.special.k0(2*np.pi*fd*Ts))
        H_set[:, :, 0] = calc_h_set(0)
        for i in range(1, int(Ns)):
            H_set[:, :, i] = H_set[:, :, i-1]*pho + calc_h_set(pho)

        self.H_set = H_set

    def init_path_loss(self, slope=0):
        """初始化路损

        路径损耗，即慢衰落。采用LTE模型。
        对于d2d接收器和基站用户而言，计算路损的方程是一致的。
        只是因为所属集合不一致，需要单独计算后再拼接。
        我们约定，考虑接收节点和发送节点的时候，都先考虑d2d，
        再考虑基站/用户。其顺序和生成坐标时的顺序一致。
        """
        # 计算距离矩阵
        n_t, m_r, n_bs, m_usr = self.n_t, self.m_r, self.n_bs, self.m_usr
        n_r_devices, n_recvs = n_t * m_r, n_t * m_r + n_bs * m_usr
        distance_matrix = np.zeros((n_recvs, n_recvs))

        def get_distances(node):
            """给定接收节点，计算所有信号的路径衰落"""
            losses = np.zeros(n_recvs)
            # d2d
            for t_index, d2d_pair in self.devices.items():
                t_device = d2d_pair['t_device']
                delta_x, delta_y = t_device.x - node.x, t_device.y - node.y
                distance = np.sqrt(delta_x**2 + delta_y**2)
                losses[t_index*m_r: t_index*m_r+m_r] = distance
            # bs
            delta_x, delta_y = self.station.x - node.x, self.station.y - node.y
            distance = np.sqrt(delta_x**2 + delta_y**2)
            losses[n_r_devices:] = distance    # 已经有n_r_devices个信道了
            return losses

        # 接收器和干扰项都先考虑d2d再考虑基站
        for t_index, d2d_pair in self.devices.items():
            r_devices = d2d_pair['r_devices']
            for r_index, r_device in r_devices.items():
                distance_matrix[t_index * self.m_r +
                                r_index] = get_distances(r_device)
        for u_index, user in self.users.items():
            distance_matrix[n_r_devices + u_index] = get_distances(user)

        self.distance_matrix = distance_matrix
        # 最小距离
        min_dis = np.concatenate((np.repeat(self.r_dev, n_r_devices),
                                  np.repeat(self.r_bs, m_usr))) \
            * np.ones((n_recvs, n_recvs))
        std = 8. + slope * (distance_matrix - min_dis)
        lognormal = np.random.lognormal(size=(n_recvs, n_recvs), sigma=std)
        path_loss = lognormal * \
            pow(10., -(120.9 + 37.6*np.log10(distance_matrix))/10.)
        self.path_loss = path_loss

    def __init__(self, n_levels, n_t_devices=9, m_r_devices=4, n_bs=1, m_usrs=4, **kwargs):
        """初始化环境"""
        # set sttributes
        self.n_t, self.m_r, self.n_bs, self.m_usr = n_t_devices, m_r_devices, n_bs, m_usrs
        self.n_recvs = self.n_t * self.m_r + self.n_bs * self.m_usr
        self.r_dev, self.r_bs, self.R_dev, self.R_bs = 0.001, 0.01, 0.1, 1
        self.Ns, self.n_levels = 50, n_levels
        self.min_power, self.max_power, self.thres_power = 5., 38., -114.  # dBm
        self.bs_power = 10 # mW
        self.m_state = 16
        self.__dict__.update(kwargs)
        # set random seed
        if 'seed' in kwargs:
            if kwargs['seed'] > 1:
                seed = kwargs['seed']
            else:
                seed = 799345
            np.random.seed(seed)
            print(f'PAEnv set random seed {seed}')

        # init attributes of pa env
        self.init_observation_space(kwargs)
        self.init_power_levels()
        self.init_pos()  # init recv pos
        self.init_jakes(Ns=self.Ns)  # init rayleigh loss using jakes model
        self.init_path_loss(slope=0)  # init path loss
        self.cur_step = 0

    def reset(self):
        self.cur_step = 0
        h_set = self.H_set[:, :, self.cur_step]
        self.loss = np.square(h_set) * self.path_loss
        return np.random.random((self.n_t * self.m_r, 3*self.m_state+2))

    def sample(self):
        sample_action = np.random.randint(0, 10, self.n_t * self.m_r)
        return sample_action

    def cal_rate(self, power, loss):
        """计算速率"""
        noise_power = 1e-3*pow(10., self.thres_power/10.)
        maxC = 1000.
        recv_power = power * loss

        signal_power = recv_power.diagonal()
        total_power = recv_power.sum(axis=1)
        inter_power = total_power - signal_power
        sinr = signal_power / (inter_power + noise_power)
        sinr = np.clip(sinr, 0, maxC)
        rate = np.log(1. + sinr)/np.log(2)

        return rate

    def get_state(self, rate, power, loss):
        """获取state

        每个接收端的state由一下三部分组成：
        1. 接收端信号中功率最大的 C 项
        2. 上个时隙的速率中对应这 C 个发送端的速率，以及自身速率
        3. 上个时隙的功率中对应这 C 个发送端的功率，以及自身功率
        其中 C 是用于控制 state 大小的可调参数，亦即 m_state
        """
        n_t, m_r, n_bs, m_usr = self.n_t, self.m_r, self.n_bs, self.m_usr
        n_recvs = n_t * m_r + n_bs * m_usr
        m_state = self.m_state
        # 检测m_state是否比n_recvs还大
        if m_state > n_recvs:
            raise ValueError(
                f"m_state({m_state}) cannot be greater than n_recvs({n_recvs})")

        # 将信号项提前
        rate_last = rate * np.ones([n_recvs, n_recvs])
        for i, _ in enumerate(rate_last):
            rate_last[i, 0], rate_last[i, i] = rate_last[i, i], rate_last[i, 0]
        power_last = power * np.ones([n_recvs, n_recvs])
        for i, _ in enumerate(power_last):
            power_last[i, 0], power_last[i, i] = power_last[i, i], \
                power_last[i, 0]
        ordered_loss = loss.copy()
        for i, _ in enumerate(ordered_loss):
            ordered_loss[i, 0], ordered_loss[i, i] = ordered_loss[i, i], \
                ordered_loss[i, 0]

        sinr_norm_inv = ordered_loss[:, 1:] / \
            np.tile(ordered_loss[:, 0:1], [1, n_recvs-1])
        sinr_norm_inv = np.log(1. + sinr_norm_inv) / \
            np.log(2)  # log representation

        sort_param = {
            'power': power_last[:, 1:],
            'rate': rate_last[:, 1:],
            'fading': sinr_norm_inv
        }

        # sorter = sinr_norm_inv
        sorter = sort_param[self.sorter]
        indices1 = np.tile(np.expand_dims(np.linspace(
            0, n_recvs-1, num=n_recvs, dtype=np.int32), axis=1), [1, m_state])
        indices2 = np.argsort(sorter, axis=1)[:, -m_state:]

        rate_last = np.hstack(
            [rate_last[:, 0:1], rate_last[indices1, indices2+1]])
        power_last = np.hstack(
            [power_last[:, 0:1], power_last[indices1, indices2+1]])
        sinr_norm_inv = sinr_norm_inv[indices1, indices2]
        metric_param = {
            'power': power_last,
            'rate': rate_last,
            'fading': sinr_norm_inv
        }
        state = np.hstack([metric_param[metric] for metric in self.metrics])

        return state[:self.n_t * self.m_r]

    def step(self, action, raw=False):
        """每个step采用一个叠加正弦波作为快衰减"""
        h_set = self.H_set[:, :, self.cur_step]
        self.loss = np.square(h_set) * self.path_loss
        if raw:
            power = action
        else:
            power = self.power_levels[action]

        # 增加BS功率项
        if len(power) == self.n_t * self.m_r:
            power = np.concatenate((power, np.full(self.m_usr, self.bs_power)))
        elif len(power) == self.n_recvs:
            power[self.n_t * self.m_r:] = np.full(self.m_usr, self.bs_power)
        else:
            msg = f"length of power should be n_recvs({self.n_recvs})" \
                f" or n_t*m_r({self.n_t*self.m_r}), but is {len(power)}"
            raise ValueError(msg)
        rate = self.cal_rate(power, self.loss)

        state = self.get_state(rate, power, self.loss)
        reward = np.sum(rate)
        done = self.cur_step == self.Ns - 1
        info = self.cur_step

        self.cur_step += 1

        return state, reward, done, info

    def render(self):
        import matplotlib.pyplot as plt

        plt.close('all')
        plt.figure(1)
        angles_circle = [i * np.pi / 180 for i in range(0, 360)]  # i先转换成double

        c_x = np.cos(angles_circle)
        c_y = np.sin(angles_circle)
        for pair in self.devices.values():
            t_d, r_ds = pair['t_device'], pair['r_devices']
            tx = plt.scatter(t_d.x, t_d.y, marker='x', label='1', s=45)
            plt.plot(t_d.x + c_x, t_d.y + c_y, 'r')  # x**2 + y**2 = 9 的圆形
            for r_d in r_ds.values():
                rx = plt.scatter(r_d.x, r_d.y, marker='o',
                                 label='2', s=25, color='orange')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.show()
