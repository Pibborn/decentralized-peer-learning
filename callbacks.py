import itertools
from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Any, Callable, Dict, List, Optional, Union, overload, SupportsInt
from itertools import combinations

import gym
import numpy as np
from peer import PeerGroup

import torch
import warnings
import os
import wandb


class PeerEvalCallback(EvalCallback):
    """
    Callback to track collective measurements about peers.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use
      ``eval_freq = max(eval_freq // n_envs, 1)``

    :param peer_group: The group of peers
    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the
        callback.
    :param log_path: Path to a folder where the evaluations
        (``evaluations.npz``) will be saved. It will be updated at each
        evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has
        not been wrapped with a Monitor wrapper)
    """

    # suboptimal but quick solution
    follow_matrix = None
    last_logged_matrix = None

    def __init__(
        self,
        peer_group: PeerGroup,
        eval_envs: List[Union[gym.Env, VecEnv]],
        n_samples=100,
        **kwargs
    ):
        self.peer_group = peer_group
        self.eval_envs = eval_envs
        self.n_samples = n_samples

        if PeerEvalCallback.follow_matrix is None:
            PeerEvalCallback.follow_matrix = np.zeros((len(peer_group),
                                                       len(peer_group)))

        super().__init__(**kwargs)

    def _on_step(self) -> bool:
        self.accumulate_followed_peers()  # needs to be done at every step
        super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # skip diversity evaluation if first epoch for any peer 
            minimum_samples_in_buffer = np.min([self.peer_group.peers[i].replay_buffer.pos for i in range(len(self.peer_group.peers))])
            if self.n_samples > minimum_samples_in_buffer:
                return True
            if 'agent_values' in self.peer_group.__dict__:
                self.track_agent_values()
            if 'trust_values' in self.peer_group.peers[0].__dict__:
                self.track_trust_values()
            PeerEvalCallback.track_followed_agent(self.peer_group.active_peer)
        return True

    def track_agent_values(self):
        n_agents = len(self.peer_group.peers)
        for i in range(n_agents):
            agent_value = self.peer_group.agent_values[i]
            wandb.log({'Peer{}_0/eval/agent_value'.format(i): agent_value},
                      commit=False)
        return True

    def track_trust_values(self):
        peer = self.peer_group.active_peer
        trust_i = self.peer_group.peers[peer].trust_values
        for j, el in np.ndenumerate(trust_i):
            wandb.log({'Peer{}_0/eval/trust_{}'.format(peer, j[0]): el},
                      commit=False)
        return True

    def accumulate_followed_peers(self):
        peer = self.peer_group.active_peer
        followed_peer = self.peer_group.peers[peer].followed_peer
        PeerEvalCallback.follow_matrix[peer, followed_peer] += 1

    @staticmethod
    def track_followed_agent(active_peer):
        if PeerEvalCallback.last_logged_matrix is None:
            diff = PeerEvalCallback.follow_matrix
        else:
            diff = PeerEvalCallback.follow_matrix -\
                   PeerEvalCallback.last_logged_matrix

        for (followed_peer,), count in np.ndenumerate(
                PeerEvalCallback.follow_matrix[active_peer]):
            wandb.log({'Peer{}_0/eval/follow_count{}'.format(
                active_peer, followed_peer): count},  commit=False)
            # also log difference
            wandb.log({'Peer{}_0/eval/follow_count_{}diff'.format(
                active_peer, followed_peer): diff[active_peer, followed_peer]},
                      commit=False)
        PeerEvalCallback.last_logged_matrix = \
            np.copy(PeerEvalCallback.follow_matrix)
