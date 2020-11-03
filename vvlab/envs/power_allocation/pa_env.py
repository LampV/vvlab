#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 2020-09-25 11:20
@edit time: 2020-11-03 11:43
@FilePath: /vvlab/vvlab/envs/power_allocation/pa_env.py
@desc: An enviornment for power allocation in d2d and BS het-nets.

Path_loss is 114.8 + 36.7*np.log10(d), follow 3GPP TR 36.873, d is
the transmitting distance, fc is 3.5GHz.
Bandwith is 20MHz, AWGN power is -114dBm, respectively.
Assume BS power is lower than 46 dBm(about 40 W).
Assume minimum transmit power: 5dBm/ maximum: 38dBm, for d2d devices.
FP algorithm, WMMSE algorithm, maximum power, random power allocation
schemes as comparisons.
downlink
"""
import scipy.special
import scipy.io
import scipy
import numpy as np
from collections import namedtuple

Node = namedtuple('Node', 'x y type')


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
        """ Initialize position of devices(DT, DR, BS and CUE).

        Establish a Cartesian coordinate system using the BS as an origin.

        Celluar User Equipment(CUE) are all located within the radius R_bs
        of the base station(typically, 1km) and outside the protected radius
        r_bs(simulation paramter, typically 0.01km) of the BS.

        The D2D transmission devices(DT) are located within the radius 
        (R_bs - R_dev) of the BS, to ensure DRs all inside the radius R_bs.
        Each DT has a cluster with sveral DRs, which is allocated within the
        radius R_dev of DT. Futher more, DRs always appear outside the raduis
        r_dev(typically 0.001km) of its DT.

        All positions are sotred with the infomation of the corresponding
        device in attributes of the environment instance, self.users and 
        self.devices.
        """
        r_bs, R_bs, r_dev, R_dev = self.r_bs, self.R_bs, self.r_dev, self.R_dev

        def random_point(min_r, radius, ox=0, oy=0):
            # https://www.cnblogs.com/yunlambert/p/10161339.html
            # renference the formulaic deduction, his code has bug at uniform
            theta = np.random.random() * 2 * np.pi
            r = np.random.uniform(min_r, radius**2)
            x, y = np.cos(theta) * np.sqrt(r), np.sin(theta) * np.sqrt(r)
            return ox + x, oy + y
        # init CUE positions
        self.station = Node(0, 0, 'station')
        self.users = {}
        for i in range(self.m_usr):
            x, y = random_point(r_bs, R_bs)
            user = Node(x, y, 'user')
            self.users[i] = user

        # init D2D positions
        self.devices = {}
        for t in range(self.n_t):
            tx, ty = random_point(r_bs, R_bs - R_dev)
            t_device = Node(tx, ty, 't_device')
            r_devices = {
                r: Node(*random_point(r_dev, R_dev, tx, ty), 'r_device')
                for r in range(self.m_r)
            }
            self.devices[t] = {'t_device': t_device, 'r_devices': r_devices}

    def init_jakes(self, fd=10, Ts=20e-3, Ns=50):
        """Initialize samples of Jakes model.

        Jakes model is a simulation of the rayleigh channel, which represents
        the small-scale fading.

        Each Rx corresponding to a (downlink) channel, each channel is a 
        source of interference to other channels. Consdering the channel 
        itself, we get a matrix representing the small-scale fading. Note that
        the interference is decided on the position of Tx and Rx, so that the
        m interferences make by m channels from the same Tx have the same 
        fading ratio.

        Args:
            fd: Doppler frequency, default 10(Hz)
            Ts: sampling period, default 20 * 1e-3(s)
            Ns: number of samples, default 50
        """
        n_t, m_r, n_bs, m_usr = self.n_t, self.m_r, 1, self.m_usr
        n_recvs = n_t * m_r + n_bs * m_usr

        def calc_h_set(pho):
            # calculate next sample of Jakes model.
            h_d2d = np.kron(np.sqrt((1.-pho**2)*0.5*(np.random.randn(n_recvs, n_t) **
                                                     2+np.random.randn(n_recvs, n_t)**2)),
                            np.ones((1, m_r), dtype=np.int32))
            h_bs = np.kron(np.sqrt((1.-pho**2)*0.5*(np.random.randn(n_recvs, n_bs)**2 +
                                                    np.random.randn(n_recvs, n_bs)**2)),
                           np.ones((1, m_usr), dtype=np.int32))
            h_set = np.concatenate((h_d2d, h_bs), axis=1)
            return h_set
        # recurrence generate all Ns samples of Jakes.
        H_set = np.zeros([n_recvs, n_recvs, int(Ns)], dtype=np.float32)
        pho = np.float32(scipy.special.k0(2*np.pi*fd*Ts))
        H_set[:, :, 0] = calc_h_set(0)
        for i in range(1, int(Ns)):
            H_set[:, :, i] = H_set[:, :, i-1]*pho + calc_h_set(pho)

        self.H_set = H_set

    def init_path_loss(self, slope=0):
        """Initialize paht loss( large-scale fading).

        The large-scale fading is related to distance. An experimental 
        formula can be used to modelling it by 3GPP TR 36.873, explained as:
        L = 36.7log10(d) + 22.7 + 26log10(fc) - 0.3(hUT - 1.5).
        When fc=3.5GHz and hUT=1.5m, the formula can be simplified to:
        L = 114.8 + 36.7*log10(d) + 10*log10(z), 
        where z is a lognormal random variable.

        As with the small-scale fading, each the n Rxs have one siginal and
        (n-1) interferences.  Using a n*n matrix to record the path loss, we
        notice that the interference from one Tx has same large-scale fading,
        Consistent with small-scale fading.
        """
        n_t, m_r, n_bs, m_usr = self.n_t, self.m_r, self.n_bs, self.m_usr
        n_r_devices, n_recvs = n_t * m_r, n_t * m_r + n_bs * m_usr

        # calculate distance matrix from initialized positions.
        distance_matrix = np.zeros((n_recvs, n_recvs))

        def get_distances(node):
            """Calculate distances from other devices to given device."""
            dis = np.zeros(n_recvs)
            # d2d
            for t_index, cluster in self.devices.items():
                t_device = cluster['t_device']
                delta_x, delta_y = t_device.x - node.x, t_device.y - node.y
                distance = np.sqrt(delta_x**2 + delta_y**2)
                dis[t_index*m_r: t_index*m_r+m_r] = distance
            # bs
            delta_x, delta_y = self.station.x - node.x, self.station.y - node.y
            distance = np.sqrt(delta_x**2 + delta_y**2)
            dis[n_r_devices:] = distance    # 已经有n_r_devices个信道了
            return dis

        # 接收器和干扰项都先考虑d2d再考虑基站
        for t_index, cluster in self.devices.items():
            r_devices = cluster['r_devices']
            for r_index, r_device in r_devices.items():
                distance_matrix[t_index * self.m_r +
                                r_index] = get_distances(r_device)
        for u_index, user in self.users.items():
            distance_matrix[n_r_devices + u_index] = get_distances(user)

        self.distance_matrix = distance_matrix

        # assign the minimum distance
        min_dis = np.concatenate(
            (np.repeat(self.r_dev, n_r_devices), np.repeat(self.r_bs, m_usr))
        ) * np.ones((n_recvs, n_recvs))
        std = 8. + slope * (distance_matrix - min_dis)
        # random initialize lognormal variable
        lognormal = np.random.lognormal(size=(n_recvs, n_recvs), sigma=std)
        
        # micro
        path_loss = lognormal * \
            pow(10., -(114.8 + 36.7*np.log10(distance_matrix))/10.)
        self.path_loss = path_loss

    def __init__(self, n_levels, n_t_devices=9, m_r_devices=4, n_bs=1, m_usrs=4, **kwargs):
        """Initialize PA environment"""
        # set sttributes
        self.n_t, self.m_r, self.n_bs, self.m_usr = n_t_devices, m_r_devices, n_bs, m_usrs
        self.n_recvs = self.n_t * self.m_r + self.n_bs * self.m_usr
        self.r_dev, self.r_bs, self.R_dev, self.R_bs = 0.001, 0.01, 0.1, 1
        self.Ns, self.n_levels = 50, n_levels
        self.min_power, self.max_power, self.thres_power = 5., 38., -114.  # dBm
        self.bs_power = 10  # W
        self.m_state = 16
        self.__dict__.update(kwargs)
        # set random seed
        if 'seed' in kwargs:
            seed = kwargs['seed'] if kwargs['seed'] > 1 else 799345
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
        return np.random.random((self.n_t * self.m_r, self.n_states))

    def sample(self):
        sample_action = np.random.randint(0, 10, self.n_t * self.m_r)
        return sample_action

    def cal_rate(self, power, loss):
        """Calculate channel rates.

        The receive power equals to emitting power times channel gain.
        The SINR can be calculated from all receive power. The channel rate
        can be infered by Shannon's formula.

        Args:
            power: emitting power.
            loss: channel gain infomation of current time slot.

        Returns:
            a vector of channel rate.
        """
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
        """Calculate and constitute the state.

        The state on each receiving end consists of the following components:
        1. channel gain of the Rx, includes siginal and interferences
        2. Tx emitting power of all channel
        3. rate of all channel

        The m_state channels (except itself) are selected by the sorter 
        assigned when initializing the observation space, default the 
        emitting power.
        So there are m_state interferance channel gain, m_state Tx emitting
        power and m_state channel rate. Notice that Tx emitting power and
        channel rate should also include the infomation of this device.
        Which parts of the metrics will consist the state is assigned when 
        initializing the observation space.

        Args:
            rate: vector of channel rate of the last time slot.
            power: vector of emitting power of the last time slot.
            loss: matrix of all channel gain of the last time slot.
        
        Returns:
            state consisted of assigned metrics ordered by assigned sorter.
        """
        # TODO power rate fading 顺序调整，同时loss重命名为fading/gain
        n_t, m_r, n_bs, m_usr = self.n_t, self.m_r, self.n_bs, self.m_usr
        n_recvs = n_t * m_r + n_bs * m_usr
        m_state = self.m_state

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
        h_set = self.H_set[:, :, self.cur_step]
        self.loss = np.square(h_set) * self.path_loss
        if raw:
            power = action
        else:
            power = self.power_levels[action]

        # append power of BS(constant)
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
        for cluster in self.devices.values():
            t_d, r_ds = cluster['t_device'], cluster['r_devices']
            tx = plt.scatter(t_d.x, t_d.y, marker='x', label='1', s=45)
            plt.plot(t_d.x + c_x, t_d.y + c_y, 'r')  # x**2 + y**2 = 9 的圆形
            for r_d in r_ds.values():
                rx = plt.scatter(r_d.x, r_d.y, marker='o',
                                 label='2', s=25, color='orange')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.show()
