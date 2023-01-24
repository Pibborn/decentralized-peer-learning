from typing import Type
import numpy as np
import torch

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from peer import make_peer_class


def make_dictator_class(cls: Type[OffPolicyAlgorithm]):
    """ Creates a mixin with the corresponding algorithm class
    :param cls: The learning algorithm (needs to have a callable critic)
    :return: The mixed in dictator agent class
    """

    class Dictator(make_peer_class(cls)):
        def __init__(self, temperature, temp_decay, algo_args, env,
                     sample_actions=True, peers_sample_with_noise=False,
                     seed=None):
            super(Dictator, self).__init__(
                peers_sample_with_noise=peers_sample_with_noise,
                temperature=temperature,
                temp_decay=temp_decay,
                algo_args=algo_args,
                solo_training=False,
                use_critic=True,
                follow_steps=1,
                seed=seed,
                env=env)
            self.sample_actions = sample_actions

        def predict(self, observation, deterministic=False, **_):
            """ The dictator always involves every party in the
            decision-making. """
            return self.get_action(observation, deterministic=deterministic)

        def critique(self, observations, actions) -> np.array:
            """ Evaluates the actions with the critic. """
            with torch.no_grad():
                a = torch.as_tensor(actions, device=self.device)
                o = torch.as_tensor(observations, device=self.device)

                # instead of using only self's critic, sum over all peers
                values = np.zeros((a.shape[0], 1), dtype=np.float32)
                for peer in self.group.peers:
                    # Compute the next Q values: min over all critic targets
                    q_values = torch.cat(peer.critic(o, a), dim=1)  # noqa
                    q_values, _ = torch.min(q_values, dim=1, keepdim=True)
                    values += q_values.cpu().numpy()
                return values

        def learn(self, _, **kwargs):
            # use again the basic learn function since we predict the action
            # always together with our peers
            return OffPolicyAlgorithm.learn(self, **kwargs)

    return Dictator


def make_weighted_dictator_class(cls: Type[OffPolicyAlgorithm]):
    """ Creates a mixin with the corresponding algorithm class
    :param cls: The learning algorithm (needs to have a callable critic)
    :return: The mixed in weighted dictator agent class
    """

    class WeightedDictator(make_dictator_class(cls)):
        def __init__(self, temperature, temp_decay, algo_args, env,
                     sample_actions=True, peers_sample_with_noise=False,
                     seed=None):
            super(WeightedDictator, self).__init__(temperature, temp_decay,
                                                   algo_args, env,
                                                   sample_actions,
                                                   peers_sample_with_noise,
                                                   seed)
            self.eval_callback = None

        def critique(self, observations, actions) -> np.array:
            values = np.empty(self.n_peers)
            for i, peer in enumerate(self.group.peers):
                if peer.eval_callback is None or \
                  len(peer.eval_callback.evaluations_results) == 0:
                    # before the first evaluation treat everyone the same
                    return np.ones(self.n_peers)
                values[i] = np.mean(peer.eval_callback.evaluations_results[-1])
            return values

        def learn(self, _, **kwargs):
            # store callback
            self.eval_callback = kwargs["callback"][0]

            super(WeightedDictator, self).learn(_, **kwargs)

        def _excluded_save_params(self):
            ex_list = super()._excluded_save_params()  # noqa
            ex_list.extend(["eval_callback"])
            return ex_list

    return WeightedDictator
