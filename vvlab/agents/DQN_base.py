#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-11-17 11:23
@edit time: 2020-05-10 22:17
@FilePath: /vvlab/agents/DQN_base.py
@desc: 创建DQN对象
"""
import torch
import torch.nn as nn
import numpy as np
from ..utils import ReplayBuffer
CUDA = torch.cuda.is_available()


class DQNBase(object):
    """The base class for DQN."""

    def __init__(self, n_states, n_actions, learning_rate=0.001,
                 discount_rate=0.0, card_no=0, **kwargs):
        """Initialize two networks and experience playback pool of DQN.

        Args:
          n_states: Number of states.
          n_actions: Number of actions.
          learning_rate: Decide how much error this time is to be learned.
          discount_rate: Attenuation value for future reward.
          card_no: Designated training card number.
          **kwargs: Incoming parameters.
        """
        # super parameters of DQN
        self.gamma = discount_rate  # future discount rate

        self.epsilon = 0.6
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.eval_every = 10
        self.card_no = card_no
        # bulid the network
        self.n_states, self.n_actions = n_states, n_actions
        self._build_net()

        self.buff_size, self.buff_thres, self.batch_size = 50000, 1000, 256
        if 'buff_size' in kwargs:
            self.buff_size = kwargs['buff_size']
        if 'buff_thres' in kwargs:
            self.buff_thres = kwargs['buff_thres']
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        self.replay_buff = ReplayBuffer(n_states, 1, buff_size=self.buff_size,
                                        buff_thres=self.buff_thres,
                                        batch_size=self.batch_size,
                                        card_no=self.card_no)
        # define optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        # record the number of steps for synchronization parameters
        self.eval_step = 0
        if CUDA:
            self.cuda()

    def _build_net(self):
        """Bulid the network.

        Raises:
          TypeError:Network build no implementation.
        """
        raise TypeError("Network build no implementation")

    def get_action(self, state):
        """Get the action at this moment.

        Args:
          state:State at this moment.

        Return:
          The action output.
        """
        # epsilon update
        self.epsilon = self.epsilon * \
            self.epsilon_decay \
            if self.epsilon > self.epsilon_min else self.epsilon
        # convert row vector to column vector
        # (1 x n_states -> n_states x 1 x 1)
        if np.random.rand() < self.epsilon:
            # random
            action_size = state.shape[0]
            return np.random.randint(0, self.n_actions, (1, action_size))
        else:
            # greedy
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_values = self.eval_net.forward(state).cpu()
            return action_values.data.numpy().argmax(axis=2)

    def get_raw_out(self, state):
        """Get the original actions.

        Args:
          state:State at this moment.

        Returns:
          Action values before selected by argmax.
        """
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_values = self.eval_net.forward(state)
        print(action_values)
        return action_values

    def add_step(self, cur_state, action, reward, done, next_state):
        """Add a record to the experience replay pool.

        Args:
          cur_state:State at this moment.
          action:Action output at this moment.
          reward:Reward after taking the action.
          done:A sign to indicate whether training is stopped.
          next_state:State at next moment.
        """
        self.replay_buff.add_step(cur_state, action, reward, done, next_state)

    def learn(self):
        """The learning process of DQN.

        Returns:
          The loss during learning process when the batch is not 0.
        """
        batch = self.replay_buff.get_batch_splited_tensor(CUDA)
        if batch is None:
            return None
        # copy parameter
        if self.eval_step % self.eval_every == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # update training steps
        self.eval_step += 1
        # split batch
        batch_cur_states, batch_actions, \
            batch_rewards, batch_dones, batch_next_states = batch
        # calculation error
        q_eval = self.eval_net(batch_cur_states)
        q_eval = q_eval.gather(1, batch_actions.long())  # shape (batch, 1)
        # detach from graph, don't backpropagate
        q_next = self.target_net(batch_next_states).detach()
        # if done, the future is not considered
        q_target = batch_rewards + self.gamma * \
            (1 - batch_dones) * \
            q_next.max(1)[0].view(
                len(batch_next_states), 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def cuda(self):
        """GPU operation using specified card."""
        self.eval_net.cuda(self.card_no)
        self.target_net.cuda(self.card_no)
