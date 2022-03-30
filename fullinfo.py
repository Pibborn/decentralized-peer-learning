import itertools as it


class PeerFullInfo:
    """ A group of agents who train separately, but share a replay buffer. """
    def __init__(self, agents):
        """
        :param agents: An iterable of peer agents
        """
        self.agents = agents

        replay_buffer_cls = self.agents[0].replay_buffer.__class__
        replay_buffer_kwargs = self.agents[0].replay_buffer_kwargs

        replay_buffer = replay_buffer_cls(100000,
                                          self.agents[0].env.observation_space, self.agents[0].env.action_space,
                                          **replay_buffer_kwargs)
        for agent in agents:
            agent.replay_buffer = replay_buffer

    def learn(self, n_epochs, max_epoch_len, callbacks, **kwargs):
        assert len(callbacks) == len(self.agents)

        for i in range(n_epochs):
            for p, peer, callback in zip(it.count(), self.agents, callbacks):
                peer.learn(total_timesteps=max_epoch_len,
                           callback=callback, tb_log_name=f"FullInfo{p}",
                           reset_num_timesteps=False, **kwargs)

