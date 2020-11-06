#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-04-06 15:36
@edit time: 2020-04-06 15:37
@FilePath: /vvlab/utils/OUProcess.py
@desc:
"""

import numpy as np


class OUProcess(object):
    """Ornstein-Uhlenbeck process"""

    def __init__(self, x_size, mu=0, theta=0.15, sigma=0.3):
        self.x = np.ones(x_size) * mu
        self.x_size = x_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def __call__(self):
        return self.noise()

    def noise(self):
        dx = self.theta * (self.mu - self.x) + \
                self.sigma * np.random.randn(self.x_size)
        self.x = self.x + dx
        return self.x
