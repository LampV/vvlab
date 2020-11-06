# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:15:30 2020

@author: Jiawei Wu
@desc: 单元测试environment
"""

from .PC_env import radio_environment
import pytest
import numpy as np


class TestEnvFunc:

    def __init__(self):
        self.seed = 2  # Seed can take any value between 0 and 49
        with pytest.warns(None) as warnings:
            self.env = radio_environment(self.seed)

    # Check that dtype is explicitly declared for gym.Box spaces
        for warning_msg in warnings:
            assert 'autodetected dtype' not in str(warning_msg.message)

        ob_space = self.env.observation_space
        act_space = self.env.action_space
        ob = self.env.reset(self.seed)
        assert ob_space.contains(
            ob), 'Reset observation: {!r} not in space'.format(ob)
        action = act_space.sample()
        observation, reward, done, _info = self.env.step(action)
        assert ob_space.contains(
            observation), 'Step observation: {!r} not in space'.format(
            observation)
        assert np.isscalar(reward), "{} is not a scalar for {}".format(
            reward, self.env)
        assert isinstance(
            done, bool), "Expected {} to be a boolean".format(done)

    def test_reset(self):
        # Reset to the same state every time.
        state_1 = self.env.reset(self.seed)
        state_2 = self.env.reset(self.seed)
        assert not (state_2 - state_1).any()

    def test_step(self):
        # This function calls almost any other function in the ns_environment.

        for action in range(self.env.num_actions):
            self.env.reset(self.seed)
            next_observation, reward, done, abort = self.env.step(action)
            assert next_observation is not None
            assert reward is not None
            assert done is not None
            assert abort is not None
