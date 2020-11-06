# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:15:30 2020

@author: Jiawei Wu
@desc: 公式测试environment
"""
from .PC_env import radio_environment
import math


class TestEnvFunc:

    def __init__(self):
        self.seed = 2  # Seed can take any value between 0 and 49
        self.env = radio_environment(self.seed)

    def test_path_loss_sub6(self):
        # The user and base station are in the same location.
        path_loss_1 = self.env._path_loss_sub6(0, 0)
        assert path_loss_1 == 0
        # The distance between user and base station is 5m.
        path_loss_2 = self.env._path_loss_sub6(3, 4)
        path_loss_2 = float(format(path_loss_2, '.6f'))
        assert path_loss_2 == 134.181195

    def test_compute_bf_vector(self):
        bf_vector_1 = self.env._compute_bf_vector(math.pi/3)
        bf_vector_2 = [(0.5+0.0j), (-1.9142843494634747e-16+0.5j),
                       (-0.5-3.8285686989269494e-16j),
                       (7.963299097640738e-16-0.5j)]
        assert not (bf_vector_2 - bf_vector_1).any()
