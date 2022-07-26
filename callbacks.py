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
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param peer_group: The group of peers
    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has notbeen
        wrapped with a Monitor wrapper)
    """

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

        self.follow_matrix = np.zeros((len(peer_group), len(peer_group)))

        super().__init__(**kwargs)

    def _on_step(self) -> bool:
        # self.accumulate_followed_peers()  # needs to be done at every step
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            # skip diversity evaluation if first epoch for any peer 
            minimum_samples_in_buffer = np.min([self.peer_group.peers[i].replay_buffer.pos for i in range(len(self.peer_group.peers))])
            if self.n_samples > minimum_samples_in_buffer:
                return True
            replay_buffers = []
            states = 100
            samples = []
            for i, peer in enumerate(self.peer_group.peers):
                num_sampled_states = states//len(self.peer_group.peers)
                sample = peer.replay_buffer.sample(num_sampled_states).observations
                samples.append(sample)
            samples = torch.cat(samples, axis=0)
            actions = []
            for peer in self.peer_group.peers:
                action, _ = peer.policy.predict(samples, deterministic=True)
                actions.append(action)
            diversity = self.track_diversity(actions)
            # self.track_followed_agent()
        return True

    def track_diversity(self, actions):
        """Computes and tracks a diversity measure between agent actions.

        Args:
            actions (np.ndarray): a 3d tensor with shape (n_agents, self.n_samples, env.action_size).
        :return: A matrix of diversity values between agents based on the L2 norm.
        """
        n_agents = len(self.peer_group.peers)
        if n_agents == 1:
            return 0
        diversity_matrix = np.zeros((n_agents, n_agents))
        for agent_1, agent_2 in combinations(range(n_agents), 2):
            diversity = np.linalg.norm(actions[agent_1]-actions[agent_2], ord=2)
            diversity_matrix[agent_1, agent_2] = diversity
            diversity_matrix[agent_2, agent_1] = diversity
            wandb.log({'Peer{}_0/eval/diversity_{}'.format(agent_1, agent_2): diversity}, commit=False)
            wandb.log({'Peer{}_0/eval/diversity_{}'.format(agent_2, agent_1): diversity}, commit=False)
        for peer_id in range(len(self.peer_group.peers)):
            wandb.log({'Peer{}_0/eval/diversity_mean'.format(peer_id): np.mean(diversity_matrix[peer_id, :])}, commit=False)
        wandb.log({'average_diversity': np.mean(diversity_matrix)})
        return diversity

    def accumulate_followed_peers(self):
        peer = self.peer_group.active_peer
        self.follow_matrix[peer, self.peer_group.peers[peer].followed_peer] \
            += 1

    def track_followed_agent(self):
        for (peer, followed_peer), count in np.ndenumerate(self.follow_matrix):
            wandb.log({'Peer{}_0/eval/follow_count{}'.format(peer,
                                                             followed_peer):
                       count},  commit=False)
