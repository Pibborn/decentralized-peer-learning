from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

class FullInfoMultiSAC(OffPolicyAlgorithm):

    def __init__(self, num_agents=4, **kwargs):
        self.agents = []
        for i in range(num_agents):
            self.agents.append(super().__init__(**kwargs))
        self._setup_shared_replay_buffer()

    def _setup_shared_replay_buffer(self):
        replay_buffer_cls = self.agents[0].__class__
        replay_buffer_kwargs = self.agents[0].replay_buffer_kwargs
        replay_buffer = replay_buffer_cls(replay_buffer_kwargs)
        for agent in self.agents:
            agent.replay_buffer = replay_buffer

    def learn(self, **kwargs):
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self



