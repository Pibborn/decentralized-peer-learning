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
        def __init__(self, temperature, temp_decay, algo_args, env_func,
                     sample_actions=True, greedy_suggestions=True):
            super(Dictator, self).__init__(temperature=temperature,
                                           temp_decay=temp_decay,
                                           algo_args=algo_args,
                                           solo_training=False,
                                           env_func=env_func,
                                           use_trust=True,
                                           follow_steps=1)
            self.sample_actions = sample_actions
            self.greedy_suggestions = greedy_suggestions

        def predict(self, observation, **_):
            """ The dictator always involves every party in the decision
            making """
            return self.get_action(observation)

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
