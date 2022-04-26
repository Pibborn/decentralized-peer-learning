import itertools as it

from peer import PeerGroup


class ParallelGroup(PeerGroup):
    def learn(self, n_epochs, max_epoch_len, callbacks, **kwargs):
        assert len(callbacks) == len(self.peers)
        # more solo epochs
        boost_single = 0 < self.switch_ratio < 1
        if boost_single:
            self.switch_ratio = 1 / self.switch_ratio

        eval_callbacks, wandb_callbacks = zip(*callbacks)

        # initialize evaluation callbacks
        for peer, callback in zip(self.peers, eval_callbacks):
            callback.eval_freq = 1  # evaluate on every call of on_step
            callback.init_callback(peer)

        for i in range(n_epochs):
            # ratio of 0 never performs a solo episode
            solo_epoch = i % (1 + self.switch_ratio) == 1
            if boost_single:
                solo_epoch = not solo_epoch

            for p, peer, callback in zip(it.count(), self.peers,
                                         wandb_callbacks):
                peer.learn(solo_epoch, total_timesteps=max_epoch_len,
                           callback=callback, tb_log_name=f"Peer{p}",
                           reset_num_timesteps=False,
                           log_interval=None, **kwargs)
                # update epoch for temperature decay
                peer.epoch += 1

            # evaluate agents
            for callback in eval_callbacks:
                # reset logger to the currently used one
                callback.logger = callback.model.logger
                # actually evaluate agents
                callback.on_step()
