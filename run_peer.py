import argparse
import datetime

import gym
import pybulletgym  # noqa
import pybullet_envs  # noqa
import numpy as np


import numpy as np
from pathlib import Path

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

import wandb
from wandb.integration.sb3 import WandbCallback

from peer import PeerGroup, make_peer_class

from utils import str2bool, add_default_values_to_parser, add_default_values_to_train_parser, new_random_seed


def add_args():
    # General
    parser = argparse.ArgumentParser(description="Peer learning.")
    parser.add_argument("--save-name", type=str, default="peer_learning")
    parser = add_default_values_to_parser(parser)

    # Training
    training = parser.add_argument_group("Training")
    training = add_default_values_to_train_parser(training)

    # Peer Learning
    peer_learning = parser.add_argument_group("Peer Learning")
    peer_learning.add_argument("--follow-steps", type=int,
                               default=1)
    peer_learning.add_argument("--switch-ratio", type=float,
                               default=1,
                               help="Ratio of peer learning episodes to solo"
                                    "episodes; 0 -> only peer learning "
                                    "episodes.")
    peer_learning.add_argument("--peer-learning", type=str2bool, nargs='?',
                               const=True, default=True)

    peer_learning.add_argument("--peers-sample-with-noise", type=str2bool, nargs='?',
                               const=True, default=True)
       
    peer_learning.add_argument("--use-agent-value", type=str2bool, nargs='?',
                               const=True, default=True)

    peer_learning.add_argument("--use-trust", type=str2bool, nargs='?',
                               const=True, default=True)
    peer_learning.add_argument("--use-trust-buffer", type=str2bool, nargs='?',
                               const=True, default=True)\

    peer_learning.add_argument("--trust-buffer-size", type=int,
                               default=1000)
    peer_learning.add_argument("--use-critic", type=str2bool, nargs='?',
                               const=True, default=True)
    peer_learning.add_argument("--trust-lr", type=float,
                               default=0.001)
    peer_learning.add_argument("--T", type=float, default=1)
    peer_learning.add_argument("--T-decay", type=float,
                               default=0)
    return parser



# environment function
def make_env(n_envs=1):
    envs = []
    for _ in range(n_envs):
        def env_func():
            env = Monitor(gym.make(args.env))
            env.seed(new_random_seed())
            return env

        envs.append(env_func)
    return DummyVecEnv(envs)


def log_reward_avg_in_wandb(callbacks):
    results = []
    for callback in callbacks:
        eval_callback = callback[0]
        result = eval_callback.evaluations_results
        print(result)
        results.append(np.mean(result))
    wandb.log({'reward_avg': np.mean(results)})


if __name__ == '__main__':
    # parse args
    parser = add_args()
    args = parser.parse_args()

    # assert if any peer learning strategy is chosen peer learning must be True
    option_on = (args.use_trust or args.use_critic or args.use_agent_value)
    assert (option_on and args.peer_learning) or not option_on

    # create results/experiments folder
    time_string = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    unique_dir = f'{time_string}__{args.job_id}'
    experiment_folder = Path.cwd().joinpath("Experiments", args.save_name,
                                            unique_dir)
    experiment_folder.mkdir(exist_ok=True, parents=True)

    # seed everything
    set_random_seed(args.seed)

    # init wandb
    wandb.tensorboard.patch(root_logdir=str(experiment_folder))
    run = wandb.init(entity='jgu-wandb', config=args.__dict__,
                     project='peer-learning',
                     monitor_gym=True, sync_tensorboard=True,
                     notes=f"Peer Learning with {args.agent_count} agents on "
                           f"the {args.env[:-3]} environment.",
                     dir=str(experiment_folder), mode=args.wandb)

    # initialize peer group
    algo_args = dict(policy="MlpPolicy", verbose=1,
                     policy_kwargs=dict(log_std_init=-3, net_arch=args.net_arch),
                     buffer_size=args.buffer_size,
                     batch_size=args.batch_size,
                     ent_coef='auto', gamma=args.gamma, tau=args.tau,
                     train_freq=args.train_freq,
                     gradient_steps=args.gradient_steps,
                     learning_starts=args.buffer_start_size, use_sde=True,
                     learning_rate=args.learning_rate,
                     tensorboard_log=str(experiment_folder),
                     device=args.device)

    peer_args = dict(temperature=args.T, temp_decay=args.T_decay,
                     algo_args=algo_args, env_func=make_env,
                     use_trust=args.use_trust, use_critic=args.use_critic,
                     buffer_size=args.trust_buffer_size,
                     follow_steps=args.follow_steps,
                     use_trust_buffer=args.use_trust_buffer,
                     solo_training=not args.peer_learning,
                     peers_sample_with_noise= args.peers_sample_with_noise)

    # create Peer classes
    SACPeer = make_peer_class(SAC)
    TD3Peer = make_peer_class(TD3)

    # create peers
    peers = []
    callbacks = []
    for i in range(args.agent_count):
        if args.mix_agents and i % 2 != 0:
            peers.append(TD3Peer(**peer_args, seed=new_random_seed()))
        else:
            peers.append(SACPeer(**peer_args, seed=new_random_seed()))

        # every agent gets its own callbacks
        callbacks.append([EvalCallback(eval_env=make_env(args.n_eval_episodes),
                                       best_model_save_path=str(experiment_folder),
                                       log_path=str(experiment_folder),
                                       eval_freq=args.eval_interval,
                                       n_eval_episodes=args.n_eval_episodes),
                          WandbCallback(gradient_save_freq=args.eval_interval,
                                        verbose=2)])

    peer_group = PeerGroup(peers, use_agent_values=args.use_agent_value,
                           lr=args.trust_lr, switch_ratio=args.switch_ratio)

    # calculate number of epochs based on episode length
    max_episode_steps = max(args.min_epoch_length,
                            gym.spec(args.env).max_episode_steps)

    # overcomes the wandb display bug
    # max_episode_steps = args.eval_interval

    n_epochs = args.steps // max_episode_steps

    # train the peer group
    peer_group.learn(n_epochs, callbacks=callbacks,
                     eval_log_path=str(experiment_folder),
                     max_epoch_len=max_episode_steps)

    log_reward_avg_in_wandb(callbacks)