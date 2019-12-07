#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-06 23:01
@edit time: 2019-12-07 21:39
@file: /test.py
"""

import torch
import numpy as np
import time
import gym
from wjwgym.agents import DDPGBase
from wjwgym.models import SimpleActorNet, SimpleCriticNet
CUDA = torch.cuda.is_available()


class DDPG(DDPGBase):
    def _build_net(self):
        n_states, n_actions = self.n_states, self.n_actions
        self.actor_eval = SimpleActorNet(n_states, n_actions, a_bound=self.bound)
        self.actor_target = SimpleActorNet(n_states, n_actions, a_bound=self.bound)
        self.critic_eval = SimpleCriticNet(n_states, n_actions)
        self.critic_target = SimpleCriticNet(n_states, n_actions)



def rl_loop():
    ENV_NAME = 'Pendulum-v0'
    RENDER = False
    MAX_EPISODES = 10000
    MAX_EP_STEPS = 200

    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(s_dim, a_dim, a_bound)
    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.add_step(s, a, r / 10, done, s_)

            if ddpg.mem_size > 10000:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300:
                    RENDER = True
                break

    print('Running time: ', time.time() - t1)


if __name__ == '__main__':
    rl_loop()
