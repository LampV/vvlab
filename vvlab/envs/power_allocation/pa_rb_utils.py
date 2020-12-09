#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2020-12-07 11:26
@FilePath: /PA/pa_rb_utils.py
@desc: 
"""

import numpy as np
import re
from collections import namedtuple


def dist(a, b):
    # a, b is Node
    # Node = namedtuple('Node', 'x y type')
    dist = np.sqrt(np.sum(
        np.square(np.array([a.x, a.y]) - np.array([b.x, b.y]))
    ))
    return dist if dist else 0.503 # if dist==0, return 0.503


def random_point_in_circle(min_r, radius, ox=0, oy=0):
    # https://www.cnblogs.com/yunlambert/p/10161339.html
    # renference the formulaic deduction, his code has bug at uniform
    theta = np.random.random() * 2 * np.pi
    r = np.random.uniform(min_r, radius**2)
    x, y = np.cos(theta) * np.sqrt(r), np.sin(theta) * np.sqrt(r)
    return ox + x, oy + y


def convert_power(power):
    Power = namedtuple('Power', 'dBm mW')
    # convert power unit
    if power.endswith(('dBm', 'dbm')):
        power_dBm = float(power[:-3])
        power_mW = np.power(10, power_dBm / 10)
    elif power.endswith(('dB', 'db')):
        power_dBm = float(power[:-2]) + 30
        power_mW = np.power(10, power_dBm / 10)
    elif power.endswith(('mW', 'mw')):
        power_mW = float(power[:-2])
        power_dBm = 10 * np.log10(power_mW / 1e-3)
    elif power.endswith(('W', 'w')):
        power_mW = 1e3 * float(power[:-1])
        power_dBm = 10 * np.log10(power_mW / 1e-3)
    else:
        power_unit = re.search(
            r'-?\d+(?:\.\d*)?(.*)$', '1.1').groups()[0]
        if power_unit:
            msg = f"Power should have unit in ('dBm', 'dB', 'mW', 'W')" \
                f", but {power_unit}."
            raise TypeError(msg)
        else:
            msg = f"Power should have unit in ('dBm', 'dB', 'mW', 'W')" \
                f", but no unit."
            raise TypeError(msg)
    return Power(power_dBm, power_mW)
