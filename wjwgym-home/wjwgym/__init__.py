#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-11-25 10:49
@edit time: 2020-03-30 10:12
"""

from gym.envs.registration import register

register(
    id='FindTreasure-v0',
    entry_point='vvlab.envs.findtreasure_env:FindTreasureEnv',
)
register(
    id='Maze-v0',
    entry_point='vvlab.envs.maze_env:MazeEnv',
)