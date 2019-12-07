#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-11-25 11:08
@edit time: 2019-12-06 23:09
@file: /wjwgym-home/setup.py
"""


from setuptools import setup, find_packages
import sys


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='wjwgym',
    version='0.1.0',
    packages=find_packages(),
    scripts=[],
    url='',
    license='MIT',
    author='Jiawei Wu',
    author_email='13260322877@163.com',
    description='OpenAI Gym test models',
    long_description='OpenAI Gym test models',
    keywords='openAI gym',
    install_requires=['numpy', 'gym', 'torch'],
    extras_require={},
)
