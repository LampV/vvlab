#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-07 20:17
@edit time: 2019-12-07 20:32
@file: /examples/dqn.py
"""
import torch
import numpy as np
import wjwgym
from functools import reduce
import gym
from wjwgym.agents import DQNBase
from wjwgym.models import SimpleDQNNet
CUDA = torch.cuda.is_available()

class DQN(DQNBase):
    def _build_net(self):
        self.eval_net = SimpleDQNNet(self.n_states, self.n_actions)
        self.target_net = SimpleDQNNet(self.n_states, self.n_actions)


def rl_loop():
    MAX_EPISODES = 200
    env = gym.make('Maze-v0')
    n_states = reduce(np.multiply, env.observation_space.shape)
    n_actions = env.action_space.n
    agent = DQN(n_states, n_actions)
    # train
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((n_states))
        done = False
        while not done:
            action = agent.get_action(cur_state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape((n_states))
            agent.add_step(cur_state, action, reward, done, next_state)
            agent.learn()
            cur_state = next_state
        print('ep: ', ep, ' steps: ', info, ' final reward: ', reward)
    print('done')
    cur_state = env.reset()
    cur_state = cur_state.reshape((n_states))
    done = False

    # test
    while not done:
        action = agent.get_action(cur_state)
        action_values = agent.get_raw_out(cur_state)
        print(action_values)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape((n_states))
        agent.add_step(cur_state, action, reward, done, next_state)
        agent.learn()
        cur_state = next_state


if __name__ == '__main__':
    rl_loop()
