#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 2020-09-25 11:20
@edit time: 2020-12-29 14:50
@file: /vvlab/vvlab/envs/power_allocation/pa_rb_env.py
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

from .pa_rb_utils import (
    dist,
    random_point_in_circle,
    convert_power
)

Node = namedtuple('Node', 'x y type')


class PAEnv:
    @property
    def n_states(self):
        """return dim of state"""
        part_len = {
            'emit_power': self.m_state * self.n_rb,
            'recv_power': self.m_state * self.n_rb,
            'rate': self.m_state,
            'csi': self.m_state,
            'emit_sum': self.m_state,
        }
        return sum(part_len[metric] for metric in self.metrics)

    @property
    def n_actions(self):
        """return num of actions"""
        return self.n_level*self.n_valid_rb

    def init_observation_space(self, kwargs):
        valid_sorters = ['recv_power', 'csi']
        # default using power
        sorter = kwargs['sorter'] if 'sorter' in kwargs else 'recv_power'
        # default using all
        valid_metrics = ['emit_power', 'recv_power', 'rate', 'csi', 'emit_sum']
        metrics = kwargs['metrics'] if 'metrics' in kwargs else valid_metrics

        # check data
        if sorter not in valid_sorters:
            msg = f'sorter should in recv_power and csi,'
            f' but is {sorter}'
            raise ValueError(msg)
        if any(metric not in valid_metrics for metric in metrics):
            msg = f'metrics should in emit_power, recv_power and rate,'
            f' but is {metrics}'
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
        random_point = random_point_in_circle

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

        Args:
            fd: Doppler frequency, default 10(Hz)
            Ts: sampling period, default 20 * 1e-3(s)
            Ns: number of samples, default 50
        """
        randn = np.random.randn
        n_tx, n_rx = self.n_tx, self.n_rx

        def calc_h_set(pho):
            _h = np.sqrt((1.-pho**2)*0.5*(randn(n_tx, n_rx)**2
                                          + randn(n_tx, n_rx)**2))
            return _h

        # recurrence generate all Ns samples of Jakes.
        pho = np.float32(scipy.special.k0(2*np.pi*fd*Ts))
        H_set = np.zeros([n_tx, n_rx, int(Ns)], dtype=np.float32)
        H_set[:, :, 0] = calc_h_set(0)
        for i in range(1, int(Ns)):
            H_set[:, :, i] = H_set[:, :, i-1]*pho + calc_h_set(pho)

        self.H_set = H_set

    def init_path_loss(self):
        """Initialize paht loss( large-scale fading).

        The large-scale fading is related to distance. An experimental
        formula can be used to modelling it by 3GPP TR 36.873, explained as:
        L = 36.7log10(d) + 22.7 + 26log10(fc) - 0.3(hUT - 1.5).
        When fc=3.5GHz and hUT=1.5m, the formula can be simplified to:
        L = 114.8 + 36.7*log10(d) + 10*log10(z),
        where z is a lognormal random variable.
        """
        n_tx, n_rx = self.n_tx, self.n_rx

        # calculate distance matrix from initialized positions.
        distance_matrix = np.zeros((n_tx, n_rx))

        devices = self.devices
        rxs = list(itertools.chain(
            (dr for c in devices.values() for dr in c['r_devices'].values()),
            (cue for cue in self.cues.values()),
            (self.station, )
        ))

        txs = list(itertools.chain(
            (c['t_device'] for c in devices.values() for _ in c['r_devices']),
            (self.station, ),
            (cue for cue in self.cues.values())
        ))

        # distance matrix
        distance_matrix = np.array([
            [dist(rx, tx) for rx in rxs]
            for tx in txs])

        self.distance_matrix = distance_matrix

        std = 4.    # std of shadow fading corresponding to lognormal
        lognormal = np.random.lognormal(size=(n_tx, n_rx), sigma=std)

        # micro
        path_loss = lognormal * \
            pow(10., -(114.8 + 36.7*np.log10(distance_matrix))/10.)
        self.path_loss = path_loss

    def init_fading(self):
        n_t, m_r, n_bs, m_cue = self.n_t, self.m_r, self.n_bs, self.m_cue
        n_tx, n_rx, n_channel = self.n_tx, self.n_rx, self.n_channel
        # 左乘行变换，复制第n_t行（BS作为Tx的行）
        cl = np.eye(n_channel, n_rx)
        for i in reversed(range(n_t, n_channel)):
            cl[i] = cl[max(i-m_cue+1, n_t)]
        # 右乘列变换，复制最后一列（BS作为Rx的列）
        cr = np.eye(n_tx, n_channel)
        cr[-1][n_tx:] = 1

        self.fading_set = {}
        for cur_step in range(self.Ns):
            h_set = self.H_set[:, :, cur_step]
            fading = np.square(h_set) * self.path_loss
            fading = np.matmul(np.matmul(cl, fading), cr)
            self.fading_set[cur_step] = fading

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
        # each DT has 1 DR, constantly
        self.n_pair, self.n_t, self.m_r = n_pair, n_pair, 1
        self.n_bs, self.m_cue = n_bs, m_cue

        # each bs-cue pair has 2 channel, uplink and downlink
        self.n_tx = self.n_t + self.n_bs + self.n_bs * self.m_cue
        self.n_rx = self.n_t * self.m_r + self.n_bs * self.m_cue + self.n_bs
        self.n_channel = self.n_t * self.m_r + self.n_bs * self.m_cue * 2

        # set random seed
        if 'seed' in kwargs:
            _seed = kwargs['seed'] if kwargs['seed'] > 1 else 799345
            np.random.seed(_seed)
            kwargs.pop('seed')
            print(f'PAEnv set random seed {_seed}')

        self.__dict__.update(kwargs)

        # check m_state
        self.m_state = min(self.n_channel, self.m_state)
        print(f"m_state: {self.m_state}")

        # set power
        _, self.bs_mW = convert_power(self.bs_power)
        _, self.cue_mW = convert_power(self.cue_power)
        self.min_dBm, self.min_mW = convert_power(self.min_power)
        self.max_dBm, self.max_mW = convert_power(self.max_power)
        _, self.noise_mW = convert_power(self.noise_power)

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
        self.init_path_loss()  # init path loss
        self.init_fading()
        self.cur_step = 0

    def reset(self):
        self.cur_step = 0
        self.fading = self.fading_set[self.cur_step]
        return np.random.random((self.n_t * self.m_r, self.n_states))

    def sample(self):
        sample_action = np.random.randint(
            0, self.n_level*self.n_valid_rb, self.n_t * self.m_r)\
            .astype(np.int32)
        return sample_action

    def get_recv_powers(self, emit_powers, fading):
        n_channel, n_rb = self.n_channel, self.n_rb
        recv_powers = np.zeros((n_channel, n_channel, n_rb))
        for i in range(n_rb):
            recv_power = emit_powers[:, :, i] * fading
            recv_powers[:, :, i] = recv_power
        return recv_powers

    def get_rates(self, recv_powers):
        maxC = 1000.
        n_channel, n_rb = self.n_channel, self.n_rb
        sinrs = np.zeros((n_channel, n_rb))
        for i in range(n_rb):
            recv_power = recv_powers[:, :, i]
            total_power = recv_power.sum(axis=0)
            signal_power = recv_power.diagonal()
            inter_power = total_power - signal_power
            _sinr = signal_power / (inter_power + self.noise_mW)
            sinrs[:, i] = np.clip(_sinr, 0, maxC)
        sinr = sinrs.sum(axis=1)
        rate = np.log(1. + sinr)/np.log(2)
        rates = rate * np.ones([n_channel, n_channel])
        return rates.T  # make rate as a column

    def get_indices(self, *metrics):
        emit_powers, recv_powers, rates, csi = metrics
        m_state = self.m_state
        # sort by recv_powers
        sort_param = {
            'recv_power': recv_powers.sum(axis=2),
            'csi': csi.copy()
        }
        sorter = sort_param[self.sorter]
        sorter[sorter == sorter.diagonal()] = float(
            'inf')    # make sure diagonal selected
        rx_indices = np.tile(np.expand_dims(np.arange(
            0, self.n_channel, dtype=np.int32), axis=0), [m_state, 1])
        tx_indices = np.argsort(sorter, axis=0)[-m_state:, :]
        return tx_indices, rx_indices

    def get_rewards(self, rates, indices):
        valid_rates = rates[indices]
        rewards = valid_rates.sum(axis=0)
        return rewards[:self.n_pair]

    def get_states(self, *metrics, indices):
        emit_powers, recv_powers, rates, csi = metrics
        # metrics
        # metrics and indices use dim_0 for tx and dim_1 for rx
        # states need dim_0 to be rx
        csi_norm = np.log2(1+csi / np.tile(np.max(csi, axis=0), [self.n_channel, 1]))
        metric_param = {
            'emit_power': np.swapaxes(emit_powers[indices], 0, 1).
            reshape(self.n_channel, -1),
            'recv_power': np.swapaxes(recv_powers[indices], 0, 1).
            reshape(self.n_channel, -1),
            'rate': np.swapaxes(rates[indices], 0, 1),
            'csi': np.swapaxes(csi_norm[indices], 0, 1),
            'emit_sum': np.swapaxes(emit_powers.sum(axis=2)[indices], 0, 1)
        }

        state = np.hstack([metric_param[metric] for metric in self.metrics])
        return state[:self.n_pair]

    def step(self, action, unit):
        if unit not in {'dBm', 'mW'}:
            msg = f"unit should in ['dBm', 'mW'], but is {unit}"
            raise ValueError(msg)

        power = self.decode_action(action, dBm=unit == 'dBm')
        csi = self.fading_set[self.cur_step]

        emit_powers = np.tile(np.expand_dims(power, axis=1),
                              (1, self.n_channel, 1))
        recv_powers = self.get_recv_powers(emit_powers, csi)
        rates = self.get_rates(recv_powers)
        metrics = emit_powers, recv_powers, rates

        indices = self.get_indices(*metrics, csi)
        rate = rates[:, 0]

        states = self.get_states(*metrics, csi, indices=indices)
        rewards = self.get_rewards(rates, indices)
        done = self.cur_step == self.Ns - 1
        info = {'step': self.cur_step, 'd2d': np.sum(rate[:self.n_pair]),
                'bs': np.sum(rate[self.n_pair: self.n_pair+self.m_cue]),
                'cue': np.sum(rate[self.n_pair+self.m_cue:]), 'rate': np.mean(rate)}

        self.cur_step += 1

        return states, rewards, done, info

    def decode_action(self, action, dBm=False):
        """decode action(especialy discrete) to RB&Power allocation."""
        n_t, m_r, n_bs, m_cue = self.n_t, self.m_r, self.n_bs, self.m_cue
        action = action.squeeze()

        # check action count
        if len(action) == self.n_t*self.m_r or len(action) == self.n_channel:
            # if action includes authorized cues, abandon
            action = action[:self.n_t*self.m_r]
        else:
            msg = f"length of action should be n_channel({self.n_channel})" \
                f" or n_pair({self.n_pair}), but is {len(action)}"
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
                        (self.n_level - 1) * (level - 1) + \
                        self.min_dBm if level else 0
                    power = str(power)+'dBm' if power else '-infdBm'
                else:
                    power = (self.max_mW - self.min_mW) / self.n_level * level
                    power = str(power)+'mW'
                alloc = {rb: convert_power(power).mW}
            else:
                msg = f"Action shape {len(action)} is not supported."
                raise ValueError(msg)

            d2d_alloc[i_dt] = alloc

        # add allocation of bs and CUE
        bs_alloc, cue_alloc = {}, {}
        for cue in range(self.m_cue):
            # channel of bs->cue use the uplink RB serial corresponding to cue
            bs_alloc[cue] = {cue: self.bs_mW}
            # cue->bs channel use downlink RB serial corresponding to cue
            # serial number in [m_cue, 2*m_cue) means downlink
            cue_alloc[cue] = {cue+self.m_cue: self.cue_mW}

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

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        pass


if __name__ == '__main__':
    env = PAEnv(10)
    env.reset()
    ret = env.step(env.sample(), unit='dBm')
    env.render()
    print(ret)
