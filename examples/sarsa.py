#!/usr/bin/env python
# coding=utf-8
"""
@create time: 2019-11-21 11:17
@author: Jiawei Wu
@edit time: 2020-09-25 16:12
@FilePath: /vvlab/examples/sarsa.py
"""

import time
import gym
import vvlab
from vvlab.agents import LinearBase


class Sarsa(LinearBase):
    """Sarsa class created based on linear QTable."""

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """Initialization of Sarsa class.

        Args:
          actions:Set of actions that can be taken.
          learning_rate:Decide how much error this time is to be learned.
          reward_decay:Attenuation value for future reward.
          e_greedy:A parameter used in decision-making to determine the proportion of actions selected according to the QTable.
        """
        super(Sarsa, self).__init__(actions, e_greedy, reward_decay, e_greedy)
        self.lr = learning_rate
        self.gamma = reward_decay

    def learn(self, s, a, r, d, s_, a_):
        """The process of updating the QTable.

        Args:
          s:State at this moment.
          a:Action at this moment.
          r:Reward after taking the action.
          d:A sign to indicate whether training is stopped.
          s_:State at next moment.
          a_:Action at next moment. 
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not d:
            # next state is not terminal
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


def rl_loop(env, agent):
    """Sarsa training process.

    Args:
      env: The environment object.
      agent: The training agent object.
    """
    for episode in range(10):
        # initial observation
        state = env.reset()

        while True:
            # fresh env
            # env.render()
            time.sleep(0.1)
            # RL choose action based on observation
            action = agent.choose_action(str(state))

            next_state, reward, done, step_count = env.step(action)

            # Sarsa
            next_action = agent.choose_action(str(next_state))
            agent.learn(str(state), action, reward, done,
                        str(next_state), next_action)

            # swap observation
            state = next_state

            # break while loop when end of this episode
            if done:
                # env.render()
                time.sleep(0.2)
                print('steps: ', step_count, 'reward: ', reward)
                break
    agent.print_table()

    # end of game
    print('game over')


if __name__ == "__main__":
    env = gym.make('Maze-v0')
    RL = Sarsa(actions=list(range(env.action_space.n)))
    rl_loop(env, RL)
