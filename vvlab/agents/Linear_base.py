#!/usr/bin/env python
# coding=utf-8
"""
@create time: 2019-11-22 10:19
@author: Jiawei Wu
@edit time: 2020-01-15 16:05
@file: /linear_agent.py
"""

import numpy as np
import pandas as pd


class LinearBase:
    """The base class for linear learning
    (QLearning, Sarsa and deformation)."""

    def __init__(self, actions: list, e_greedy: float,
                 learning_rate, reward_decay):
        """Initialize the base class.

        Args:
          actions: Set of actions that can be taken.
          e_greedy: Initial value of epsilon-greedy.
          learning_rate: Decide how much error this time is to be learned.
          reward_decay: Attenuation value for future reward.
        """
        self.actions = actions  # a list
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.lr = learning_rate
        self.gamma = reward_decay

    def choose_action(self, observation, epsilon=None):
        """Get the action corresponding to the current state.

        Select the value of the largest value from the Q table
        according to the probability of epsilon, and randomly select the rest.

        Args:
          observation: State at this moment.
          epsilon: The epsilon value, if this value is not passed, it will be
          random according to the default value set at the beginning.

        Returns:
          Action at this moment.
        """
        # make sure epsilon is set
        if not epsilon:
            epsilon = self.epsilon
        # check whether the current state exists
        # (it will be initialized if it does not exist)
        self.check_state_exist(observation)

        # select action
        if np.random.uniform() < epsilon:
            # get the QTable row corresponding to the current state
            state_actions = self.q_table.loc[observation, :]
            # randomly select from all indexes with the largest value
            action = \
                np.random.choice(
                    state_actions[state_actions ==
                                  np.max(state_actions)].index)
        else:
            # random selection
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        """Check whether the status exists.

        If it does not exist,
        add it to the row of QTable and initialize it to all 0s.

        Args:
          state:State.
        """
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def print_table(self):
        """Print QTable content."""
        q_table = self.q_table
        print(q_table)
