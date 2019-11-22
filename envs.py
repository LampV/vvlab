#!/usr/bin/env python
# coding=utf-8
"""
@create time: 2019-11-22 15:08
@author: Jiawei Wu
@edit time: 2019-11-22 15:08
@file: /rltest/envs.py
"""


from find_treasure import find_treasure_env

def make(env_name):
    if env_name == 'FindTreasure-v0':
        return find_treasure_env()