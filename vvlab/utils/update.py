#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-04-07 19:52
@edit time: 2020-04-07 19:54
@FilePath: /vvlab/utils/update.py
@desc: update funcs
"""


def soft_update(target, source, tau):
    """使用tau作为系数的软更新
    使用公式 param_{target} = (1 - tau) * param_{target} + tau * param_{source} 更新

    @param target: 更新的目标网络
    @param source: 更新的源网络
    @param tau: 更新系数
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
