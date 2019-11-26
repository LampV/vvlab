#!/usr/bin/env python
# coding=utf-8
"""
@create time: 2019-11-21 11:17
@author: Jiawei Wu
@edit time: 2019-11-26 10:41
@file: /test.py
"""

import time
from linear_agent import QLearning, Sarsa
import gym 
import wjwgym

def update(env, agent):
    """
    @description: 
    @param env: 传入的环境对象
    @param agent: 传入的智能体对象 
    @return: 
    """
    for episode in range(20):
        # initial observation
        state = env.reset()

        while True:
            # fresh env
            # env.render()
            time.sleep(0.1)
            # RL choose action based on observation
            action = agent.choose_action(str(state))

            # RL take action and get next observation and reward
            next_state, reward, done, step_count = env.step(action)
            # RL learn from this transition
            next_action = agent.choose_action(str(next_state))
            agent.learn(str(state), action, reward, str(next_state), next_action)
            # agent.print_table()
            # swap observation
            state = next_state

            # break while loop when end of this episode
            if done:
                # env.render()
                time.sleep(0.2)
                print('steps: ', step_count)
                break

    # end of game
    print('game over')

if __name__ == "__main__":
    enviornment = gym.make('MyGym-v0')
    RL = Sarsa(actions=list(range(enviornment.action_space.n)))
    update(enviornment, RL)
