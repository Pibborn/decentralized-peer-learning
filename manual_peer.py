import numpy as np

from abc import ABC

from suggestionbuffer import SuggestionBuffer


class ManualPeer(ABC):
    def __init__(self, seed=None, **_):
        self.followed_peer = None
        self.n_peers = None
        self.group = None
        self.peer_values = dict()
        self.peer_value_functions = dict()
        self.buffer = SuggestionBuffer(0)
        self.epoch = 0
        self.policy = self  # for suggestions
        self.np_random = np.random.default_rng(seed)

    def learn(self, *_, **_1):
        self.followed_peer = np.where(self.group.peers == self)
        pass  # don't learn

    def _predict(self, obs, **kwargs):
        # needs to be overwritten in subclasses
        raise NotImplementedError

    def predict(self, obs, **kwargs):
        # handle VecEnvs
        actions = [self._predict(o, **kwargs) for o in np.atleast_2d(obs)]
        return np.array(actions), None
