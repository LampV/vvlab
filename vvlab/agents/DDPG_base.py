#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-04 10:36
@edit time: 2020-04-07 19:55
@FilePath: /vvlab/agents/DDPG_base.py
"""
import os
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from ..utils import CUDA, ReplayBuffer
from ..utils.update import soft_update
import warnings


class DDPGBase(object):
    """The base class for DDPG."""

    def __init__(self, n_states, n_actions, action_bound=1, buff_size=1000,
                 buff_thres=0, batch_size=32, lr_a=0.001, lr_c=0.002, tau=0.01,
                 gamma=0.9, summary=False, card_no=0, *args, **kwargs):
        """Initialize the base class.

         Args:
           n_states:Number of states.
           n_actions:Number of actions.
           action_bound:Limit the changes of actions.
           buff_size:Size of experience replay pool.
           buff_thres:The minimum number of experience replay pool outputs.
           batch_size:The size of a batch.
           lr_a:Learning rate of actor.
           lr_c:Learning rate of critic.
           tau:Soft update factor.
           gamma:Set the weight of the value of the next action.
           summary:To decide whether provide the possibility
           to obtain the file save path from the parameter.
           *args:Pack the parameters into tuples and call the function body.
           **kwargs:Pack the parameters into dicts and call the function body.
         """
        # compatible parameters
        # TODO remove parameter compatibility in 0.3.0
        if 'bound' in kwargs:
            warnings.warn("'bound' is deprecated and will remove after 0.3.0. "
                          "Use 'action_bound' instead.",
                          DeprecationWarning, stacklevel=2)
            action_bound = kwargs['bound']
            self.bound = action_bound
        if 'exp_size' in kwargs:
            warnings.warn("'exp_size' is deprecated \
                    and will remove after 0.3.0. "
                          "Use 'buff_size' instead.",
                          DeprecationWarning, stacklevel=2)
            buff_size = kwargs['exp_size']
            self.exp_size = buff_size
        if 'exp_thres' in kwargs:
            warnings.warn("'exp_thres' is deprecated \
                    and will remove after 0.3.0. "
                          "Use 'buff_thres' instead.",
                          DeprecationWarning, stacklevel=2)
            buff_thres = kwargs['exp_thres']
            self.exp_thres = buff_thres

        # copy parameters
        self.n_states, self.n_actions, self.action_bound = \
            n_states, n_actions, action_bound
        self.buff_size, self.buff_thres, self.batch_size = \
            buff_size, buff_thres, batch_size
        self.lr_a, self.lr_c, self.tau, self.gamma = lr_a, lr_c, tau, gamma
        self.card_no = card_no
        self.summary = summary
        self.kwargs = kwargs

        # initialize episode and step
        self.episode, self.step = 0, 0
        # parameter override
        self._param_override()

        # create experience replay pool
        self.buff = ReplayBuffer(self.n_states, self.n_actions,
                                 buff_size=self.buff_size,
                                 buff_thres=self.buff_thres,
                                 card_no=self.card_no)

        # bulid neural networks
        self._build_net()
        # specify optimizer
        self.actor_optim = \
            torch.optim.Adam(self.actor_eval.parameters(), lr=self.lr_a)
        self.critic_optim = \
            torch.optim.Adam(self.critic_eval.parameters(), lr=self.lr_c)
        # specify loss function
        self.mse_loss = nn.MSELoss()

        # specify noise generator
        self._build_noise()

        # specify summary writer
        self._build_summary_writer()

        # start cuda
        if CUDA:
            self.cuda()

    def _param_override(self):
        """A method for subclass to override the parameters of the baseclass.

        For example: Modify whether summary is enabled.
        This method should be used with caution.
        """
        pass

    def _build_net(self):
        """Build network.

        Raises:
          TypeError:Network build no implementation.
        """
        raise TypeError("网络构建函数未被实现")

    def _build_noise(self, *args):
        """Build noise generator.

        Raises:
          TypeError:Noise generator build no implementation.
        """
        raise TypeError("噪声发生器构建函数未被实现")

    def _build_summary_writer(self):
        """Build summary writer.

        When "summary" is set to true,
        if no summary writer is specified, it will be set to None,
        if the save path is specified, use the save path,
        otherwise use the default path.
        """
        if self.summary:
            if 'summary_path' in self.kwargs:
                self.summary_writer = \
                    SummaryWriter(log_dir=self.kwargs['summary_path'])
                self._build_summary_writer(self.kwargs['summary_path'])
            else:
                self.summary_writer = SummaryWriter()
        else:
            self.summary_writer = None

    def get_summary_writer(self):
        """Get summary writer."""
        return self.summary_writer

    def _get_action(self, s):
        """Get the selected action under current state.

        Args:
          s:Given state.
        Returns:
          The selected action.
        """
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.actor_eval.forward(s).detach().cpu().numpy()
        return action

    def get_action(self, s):
        """Get the selected action under current state.

        Args:
          s:Given state.

        Returns:
          The selected action.
        """
        return self._get_action(s)

    def _save(self, save_path, append_dict={}):
        """Save the network parameters of the current model.

        Args:
          save_path: The save path of the model.
          append_dict: What needs to be saved in addition to the network model.
        """
        states = {
            'actor_eval_net': self.actor_eval.state_dict(),
            'actor_target_net': self.actor_target.state_dict(),
            'critic_eval_net': self.critic_eval.state_dict(),
            'critic_target_net': self.critic_target.state_dict(),
        }
        states.update(append_dict)
        torch.save(states, save_path)

    def save(self, episode=None, save_path='./cur_model.pth'):
        """The saved default implementation.

        Args:
          episode: Current episode.
          save_path: The save path of the model，default is'./cur_model.pth'.
        """
        append_dict = {
            'episode': self.episode if episode is None else episode,
            'step': self.step
        }
        self._save(save_path, append_dict)

    def _load(self, save_path):
        """Load the model parameter.

        Args:
          save_path: The save path of the model.

        Returns:
          The loaded model dictionary.
        """
        if CUDA:
            states = torch.load(save_path, map_location=torch.device('cuda'))
        else:
            states = torch.load(save_path, map_location=torch.device('cpu'))

        # load network parameters from the model
        self.actor_eval.load_state_dict(states['actor_eval_net'])
        self.actor_target.load_state_dict(states['actor_target_net'])
        self.critic_eval.load_state_dict(states['critic_eval_net'])
        self.critic_target.load_state_dict(states['critic_target_net'])

        # load 'episode' and 'step' from the model
        self.episode, self.step = states['episode'], states['step']
        # return states
        return states

    def load(self, save_path='./cur_model.pth'):
        """The default implementation of the loaded model.

        Args:
          save_path: The save path of the model，default is'./cur_model.pth'.

        Returns:
          Recorded episode value.
        """
        print('\033[1;31;40m{}\033[0m'.format('加载模型参数...'))
        if not os.path.exists(save_path):
            print('\033[1;31;40m{}\033[0m'.format('没找到保存文件'))
            return -1
        else:
            states = self._load(save_path)
            return states['episode']

    def _learn(self):
        """Train network.

        Returns:
          The td_error and loss of training process.
        """
        # assign eval network parameters to target network
        soft_update(self.actor_target, self.actor_eval, self.tau)
        soft_update(self.critic_target, self.critic_eval, self.tau)

        # get and spilt the batch
        batch = self.buff.get_batch_splited_tensor(CUDA, self.batch_size)
        if batch is None:
            return None, None
        else:
            self.start_train = True
        batch_cur_states, batch_actions, batch_rewards, \
            batch_dones, batch_next_states = batch

        # calculate target_q, guide cirtic update
        # calculate next_action that
        # the target network will choose through a_target and next_state;
        # compute the q_values of the next state through
        # target_q, next_states, and next_actions just calculated
        target_q_next = \
            self.critic_target(batch_next_states,
                               self.actor_target(batch_next_states))
        target_q = \
            batch_rewards + self.gamma * \
            (1 - batch_dones) * \
            target_q_next   # If done, the future is not considered.
        # guide critic update
        q_value = self.critic_eval(batch_cur_states, batch_actions)
        td_error = self.mse_loss(target_q, q_value)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()

        # guide actor update
        policy_loss = \
            self.critic_eval(batch_cur_states, self.actor_eval(
                batch_cur_states))  # 用更新的eval网络评估这个动作
        # If a is a correct behavior, then its policy_loss
        # should be closer to 0.
        loss_a = -torch.mean(policy_loss)
        self.actor_optim.zero_grad()
        loss_a.backward()
        self.actor_optim.step()
        return td_error.detach().cpu().numpy(), loss_a.detach().cpu().numpy()

    def learn(self):
        """Specify the trained loss as the data required for visualization."""
        c_loss, a_loss = self._learn()
        if all((c_loss is not None, a_loss is not None)):
            self.step += 1
            if self.summary_writer:
                self.summary_writer.add_scalar('c_loss', c_loss, self.step)
                self.summary_writer.add_scalar('a_loss', a_loss, self.step)

    def _add_step(self, s, a, r, d, s_):
        """Add a record to the experience replay pool.

        Args:
          s:State at this moment.
          a:Action output at this moment.
          r:Reward after taking the action.
          d:A sign to indicate whether training is stopped.
          s_:State at next moment.
        """
        self.buff.add_step(s, a, r, d, s_)

    def add_step(self, s, a, r, d, s_):
        """The default implementation of adding records, do nothing except add records.

        Args:
          s:State at this moment.
          a:Action output at this moment.
          r:Reward after taking the action.
          d:A sign to indicate whether training is stopped.
          s_:State at next moment.
        """
        self._add_step(s, a, r, d, s_)

    def cuda(self):
        """Use gpu training."""
        self.actor_eval.cuda(self.card_no)
        self.actor_target.cuda(self.card_no)
        self.critic_eval.cuda(self.card_no)
        self.critic_target.cuda(self.card_no)
