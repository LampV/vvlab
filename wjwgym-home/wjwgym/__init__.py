#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-11-25 10:49
@edit time: 2019-12-06 23:19
@file: /wjwgym/wjwgym/__init__.py
"""

from wjwgym.agents import DDPGBase
from gym.envs.registration import register

register(
    id='FindTreasure-v0',
    entry_point='wjwgym.envs.findtreasure_env:FindTreasureEnv',
)
register(
    id='Maze-v0',
    entry_point='wjwgym.envs.maze_env:MazeEnv',
)