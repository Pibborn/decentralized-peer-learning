import itertools as it


class PeerFullInfo:
    """ A group of agents who train separately, but share a replay buffer. """

    def __init__(self, agents):
        """
        :param agents: An iterable of peer agents
        """
        self.agents = agents

        # copy values from first agent's buffer
        first_buffer = self.agents[0].replay_buffer
        replay_buffer_cls = first_buffer.__class__
        replay_buffer_kwargs = self.agents[0].replay_buffer_kwargs
        replay_buffer = replay_buffer_cls(first_buffer.buffer_size,
                                          first_buffer.observation_space,
                                          first_buffer.action_space,
                                          device=first_buffer.device,
                                          n_envs=first_buffer.n_envs,
                                          **replay_buffer_kwargs)
        for agent in agents:
            agent.replay_buffer = replay_buffer

    def learn(self, n_epochs, max_epoch_len, callbacks, **kwargs):
        assert len(callbacks) == len(self.agents)

        for i in range(n_epochs):
            for p, peer, callback in zip(it.count(), self.agents, callbacks):
                peer.learn(total_timesteps=max_epoch_len,
                           callback=callback, tb_log_name=f"Peer{p}",
                           reset_num_timesteps=False, **kwargs)
