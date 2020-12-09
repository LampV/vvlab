#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 2020-09-25 11:20
@edit time: 2020-12-09 22:31
@FilePath: /vvlab/vvlab/envs/power_allocation/pa_rb_env.py
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
from collections import namedtuple
from pathlib import Path
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy.special

import pa_rb_utils as utils

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
        return self.n_level

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
        device in attributes of the environment instance, self.cues and
        self.devices.
        """
        random_point = utils.random_point_in_circle

        r_bs, R_bs, r_dev, R_dev = self.r_bs, self.R_bs, self.r_dev, self.R_dev
        # init CUE positions
        self.station = Node(0, 0, 'station')
        self.cues = {}
        for i in range(self.m_cue):
            x, y = random_point(r_bs, R_bs)
            cue = Node(x, y, 'cue')
            self.cues[i] = cue

        # init D2D positions
        self.devices = {}
        for t in range(self.n_t):
            tx, ty = random_point(r_bs, R_bs - R_dev)
            t_device = Node(tx, ty, 't_device')
            r_devices = {
                r: Node(*random_point(r_dev, R_dev, tx, ty), 'r_devices')
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
        randn = np.random.randn

        n_t, m_r, n_bs, m_cue = self.n_t, self.m_r, 1, self.m_cue
        n_channel = self.n_channel

        def foo(pho, n, m):
            # each channel have n_channel h to other channels(include self)
            # each Tx send m signal, the number of this kind of Tx is n
            _h = np.sqrt((1.-pho**2)*0.5*(randn(n_channel, n)**2
                                          + randn(n_channel, n)**2))
            return np.kron(_h, np.ones((1, m), dtype=np.int32))

        def calc_h_set(pho):
            # calculate next sample of Jakes model.
            return np.concatenate((foo(pho, n_t, m_r),
                                   foo(pho, n_bs, m_cue),
                                   foo(pho, m_cue, n_bs)
                                   ), axis=1)
        # recurrence generate all Ns samples of Jakes.
        pho = np.float32(scipy.special.k0(2*np.pi*fd*Ts))
        H_set = np.zeros([n_channel, n_channel, int(Ns)], dtype=np.float32)
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
        n_t, m_r, n_bs, m_cue = self.n_t, self.m_r, self.n_bs, self.m_cue
        n_channel = self.n_channel

        # calculate distance matrix from initialized positions.
        distance_matrix = np.zeros((n_channel, n_channel))

        devices = self.devices
        rxs = list(itertools.chain(
            (dr for c in devices.values() for dr in c['r_devices'].values()),
            (cue for cue in self.cues.values()),
            (self.station for _ in self.cues)
        ))

        txs = list(itertools.chain(
            (c['t_device'] for c in devices.values() for _ in c['r_devices']),
            (self.station for _ in self.cues),
            (cue for cue in self.cues.values())
        ))

        # TODO  检测distance和h_set的轴是否一致
        # distance matrix
        distance_matrix = np.array([
            [utils.dist(rx, tx) for rx in rxs]
            for tx in txs])

        self.distance_matrix = distance_matrix

        std = 4.    # std of shadow fading corresponding to lognormal
        lognormal = np.random.lognormal(size=(n_channel, n_channel), sigma=std)

        # micro
        path_loss = lognormal * \
            pow(10., -(114.8 + 36.7*np.log10(distance_matrix))/10.)
        self.path_loss = path_loss

    def __init__(self, n_level,
                 n_pair=9, n_bs=1, m_cue=4, **kwargs):
        """Initialize PA environment"""
        # constant attributes
        self.m_state = 16
        self.r_dev, self.r_bs, self.R_dev, self.R_bs = 0.001, 0.01, 0.1, 1
        self.Ns = 50 if 'Ns' not in kwargs else kwargs['Ns']
        self.bs_power, self.cue_power = '10W', '1W'  # 10W and 1W, respectively
        self.min_power, self.max_power, self.noise_power = '5dBm', '38dBm', '-114dBm'

        # set attributes
        self.n_level = n_level
        self.n_t, self.m_r = n_pair, 1  # each DT has 1 DR, constantly
        self.n_bs, self.m_cue = n_bs, m_cue

        # each bs-cue pair has 2 channel, uplink and downlink
        self.n_channel = self.n_t * self.m_r + self.n_bs * self.m_cue * 2

        self.__dict__.update(kwargs)

        # set random seed
        if 'seed' in kwargs:
            seed = kwargs['seed'] if kwargs['seed'] > 1 else 799345
            np.random.seed(seed)
            print(f'PAEnv set random seed {seed}')

        # set power
        _, self.bs_power = utils.convert_power(self.bs_power)
        _, self.cue_power = utils.convert_power(self.cue_power)
        self.min_dBm, self.min_mW = utils.convert_power(self.min_power)
        self.max_dBm, self.max_mW = utils.convert_power(self.max_power)
        _, self.noise_mW = utils.convert_power(self.noise_power)

        # set rb
        self.n_rb = 2 * self.n_bs * self.m_cue
        if ('rb' in kwargs and kwargs['rb'] == 'duplex'):
            self.n_valid_rb = 2 * self.n_bs * self.m_cue
        else:
            self.n_valid_rb = self.n_bs * self.m_cue

        # init attributes of pa env
        self.init_observation_space(kwargs)
        self.init_pos()  # init recv pos
        self.init_jakes(Ns=self.Ns)  # init rayleigh loss using jakes model
        self.init_path_loss(slope=0)  # init path loss
        self.cur_step = 0

    def reset(self):
        self.cur_step = 0
        h_set = self.H_set[:, :, self.cur_step]
        self.fading = np.square(h_set) * self.path_loss
        return np.random.random((self.n_t * self.m_r, self.n_states))

    def sample(self):
        sample_action = np.random.randint(
            0, self.n_level*self.n_valid_rb, self.n_t * self.m_r).astype(np.int32)
        return sample_action

    def cal_rate(self, power, fading):
        """Calculate channel rates.

        The receive power equals to emitting power times channel gain.
        The SINR can be calculated from all receive power. The channel rate
        can be infered by Shannon's formula.

        Args:
            power: emitting power.
            fading: channel gain infomation of current time slot.

        Returns:
            a vector of channel rate.
        """
        power = power * np.ones((self.n_channel, self.n_channel, self.n_rb))
        maxC = 1000.
        sinrs = np.zeros((self.n_channel, self.n_rb))
        for i in range(self.n_rb):
            recv_power = power[:,:,i] * fading
            signal_power = recv_power.diagonal()
            total_power = recv_power.sum(axis=1)
            inter_power = total_power - signal_power
            # if no signal, no interference
            _sinr = signal_power / (inter_power + self.noise_mW)
            sinrs[:, i] = np.clip(_sinr, 0, maxC)

        sinr = sinrs.sum(axis=1)
        rate = np.log(1. + sinr)/np.log(2)

        return rate

    def get_state(self, power, rate, fading):
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
            power: vector of emitting power of the last time slot.
            rate: vector of channel rate of the last time slot.
            fading: matrix of all channel gain of the last time slot.

        Returns:
            state consisted of assigned metrics ordered by assigned sorter.
        """
        n_t, m_r, n_bs, m_cue = self.n_t, self.m_r, self.n_bs, self.m_cue
        n_channel = self.n_channel
        m_state = self.m_state

        if m_state > n_channel:
            msg = f"m_state should be less than n_channel({n_channel})" \
                f", but was {m_state}"
            raise ValueError(msg)

        # 将信号项提前
        rate_last = rate * np.ones([n_channel, n_channel])
        for i, _ in enumerate(rate_last):
            rate_last[i, 0], rate_last[i, i] = rate_last[i, i], rate_last[i, 0]
        power_last = power.sum(axis=1) * np.ones([n_channel, n_channel])
        for i, _ in enumerate(power_last):
            power_last[i, 0], power_last[i, i] = power_last[i, i], \
                power_last[i, 0]
        ordered_fading = fading.copy()
        for i, _ in enumerate(ordered_fading):
            ordered_fading[i, 0], ordered_fading[i,
                                                 i] = ordered_fading[i, i], ordered_fading[i, 0]

        sinr_norm_fading = ordered_fading[:, 1:] / \
            np.tile(ordered_fading[:, 0:1], [1, n_channel-1])
        sinr_norm_fading = np.log2(1. + sinr_norm_fading)

        sort_param = {
            'power': power_last[:, 1:],
            'rate': rate_last[:, 1:],
            'fading': sinr_norm_fading
        }

        # sorter = sinr_norm_inv
        sorter = sort_param[self.sorter]
        indices1 = np.tile(np.expand_dims(np.linspace(
            0, n_channel-1, num=n_channel, dtype=np.int32), axis=1), [1, m_state])
        indices2 = np.argsort(sorter, axis=1)[:, -m_state:]

        rate_last = np.hstack(
            [rate_last[:, 0:1], rate_last[indices1, indices2+1]])
        power_last = np.hstack(
            [power_last[:, 0:1], power_last[indices1, indices2+1]])
        sinr_norm_fading = sinr_norm_fading[indices1, indices2]
        metric_param = {
            'power': power_last,
            'rate': rate_last,
            'fading': sinr_norm_fading
        }
        state = np.hstack([metric_param[metric] for metric in self.metrics])

        return state[:self.n_t * self.m_r]

    def decode_action(self, action, dBm=False):
        """decode action(especialy discrete) to RB&Power allocation."""
        n_t, m_r, n_bs, m_cue = self.n_t, self.m_r, self.n_bs, self.m_cue
        action = action.squeeze()
        # check action count
        if len(action) == self.n_t*self.m_r or len(action) == self.n_channels:
            # if action includes authorized cues, abandon
            action = action[:self.n_t*self.m_r]
        else:
            msg = f"length of action should be n_channel({self.n_channel})" \
                f" or n_t({self.n_t}), but is {len(action)}"
            raise ValueError(msg)

        d2d_alloc = {}
        for i_dt, a in enumerate(action):
            if a.dtype in [np.float32, np.float64]:
                # Continuous action, direct to power of mW
                alloc = {rb: power for rb, power in enumerate(a) if power > 0}
            elif a.dtype in [np.int32, np.int64]:
                # Discrete action, need convert
                rb, level = divmod(a, self.n_level)
                if dBm:
                    power = (self.max_dBm - self.min_dBm) / \
                        self.n_level * level
                    power = str(power)+'dBm' if power else '-infdBm'
                else:
                    power = (self.max_mW - self.min_mW) / self.n_level * level
                    power = str(power)+'mW'
                alloc = {rb: utils.convert_power(power).mW}
            else:
                msg = f"Action shape {len(action)} is not supported."
                raise ValueError(msg)

            d2d_alloc[i_dt] = alloc

        # add allocation of bs and CUE
        bs_alloc, cue_alloc = {}, {}
        for cue in range(self.m_cue):
            # channel of bs->cue use the uplink RB serial corresponding to cue
            bs_alloc[cue] = {cue: self.bs_power}
            # cue->bs channel use downlink RB serial corresponding to cue
            # serial number in [m_cue, 2*m_cue) means downlink
            cue_alloc[cue] = {cue+self.m_cue: self.cue_power}

        allocation_map = {'bs': bs_alloc, 'cue': cue_alloc, 'd2d': d2d_alloc}
        self.allocation_map = allocation_map

        allocations = np.zeros((self.n_channel, self.n_rb))
        for d_index, alloc in allocation_map.get('d2d').items():
            for rb, power in alloc.items():
                allocations[d_index][rb] = power
        for b_index, alloc in allocation_map.get('bs').items():
            for rb, power in alloc.items():
                allocations[n_t*m_r + b_index][rb] = power
        for c_index, alloc in allocation_map.get('cue').items():
            for rb, power in alloc.items():
                allocations[n_t*m_r + n_bs*m_cue + c_index][rb] = power
        return allocations

    def step(self, action, dBm=False):
        h_set = self.H_set[:, :, self.cur_step]
        self.fading = np.square(h_set) * self.path_loss
        self.allocations = self.decode_action(action, dBm=dBm)

        rate = self.cal_rate(self.allocations, self.fading)

        state = self.get_state(self.allocations, rate, self.fading)
        reward = np.sum(rate)
        done = self.cur_step == self.Ns - 1
        info = self.cur_step

        self.cur_step += 1

        return state, reward, done, info

    def render(self):

        def cir_edge(center, radius, color):
            patch = mpatches.Circle(center, radius,
                                    fc='white', ec=color, ls='--')
            return patch

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        # draw d2d pairs
        for t_idx, pair in self.devices.items():
            t, rs = pair['t_device'], pair['r_devices']
            # draw edge
            ax.add_patch(cir_edge((t.x, t.y), self.R_dev, 'green'))
            # draw t device
            ax.scatter([t.x], [t.y], marker='s', s=100, c='green', zorder=10)
            # draw r devices
            for _, r in rs.items():
                ax.scatter([r.x], [r.y], marker='o',
                           s=60, c='green', zorder=10)

        # draw cell and bs
        cell_xs = self.R_bs * \
            np.array([0, np.sqrt(3)/2, np.sqrt(3)/2,
                      0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0])
        cell_ys = self.R_bs * np.array([1, .5, -.5, -1, -.5, .5, 1])
        ax.plot(cell_xs, cell_ys, color='black')

        ax.scatter([0.0], [0.0], marker='^', s=100, c='blue', zorder=30)
        # draw usrs
        for _, usr in self.cues.items():
            ax.scatter([usr.x], [usr.y], marker='x',
                       s=100, c='orange', zorder=20)
            ax.plot([0, usr.x], [0, usr.y], ls='--', c='blue', zorder=20)

        save_path = Path('pa_env.png')
        plt.savefig(save_path)
        plt.close(fig)
        return save_path


if __name__ == '__main__':
    env = PAEnv(10, n_pair=9, m_cue=4)
    env.reset()
    ret = env.step(env.sample())
    env.render()
    print(ret)
