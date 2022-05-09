import argparse
import datetime

import gym
import pybulletgym  # noqa
import pybullet_envs  # noqa

import numpy as np
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

import wandb
from wandb.integration.sb3 import WandbCallback

from fullinfo import PeerFullInfo
from fullinfo import FullInfoMultiThreading

from utils import str2bool,add_default_values_to_parser, add_default_values_to_train_parser, new_random_seed


def add_args():
    # create arg parser
    parser = argparse.ArgumentParser(description="Full Information")
    # General
    parser.add_argument("--save-name", type=str, default="full_info")

    parser = add_default_values_to_parser(parser)

    # Training
    training = parser.add_argument_group("Training")
    training = add_default_values_to_train_parser(training)

    parser.add_argument("--track-video", action='store_true')

    parser.add_argument("-t", "--multi-threading", action="store_true",
                        help="Run agents parallel in different threads.")


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


def create_sac_agents(env, num_agents):
    agent_list = []
    for i in range(num_agents):
        agent = SAC(policy="MlpPolicy", env=env, verbose=1,
                    policy_kwargs=dict(log_std_init=-3,
                                       net_arch=args.net_arch),
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    ent_coef='auto', gamma=args.gamma, tau=args.tau,
                    train_freq=args.train_freq,
                    gradient_steps=args.gradient_steps,
                    learning_starts=args.buffer_start_size, use_sde=True,
                    learning_rate=args.learning_rate,
                    tensorboard_log=str(experiment_folder),
                    seed=new_random_seed(),
                    device=args.device)
        agent_list.append(agent)
    return agent_list


def train_single(agent, env_test, log_interval, savedir):
    eval_callback = EvalCallback(env_test,
                                 best_model_save_path=savedir,
                                 log_path=savedir,
                                 eval_freq=log_interval,
                                 deterministic=True, render=False)
    wandb_callback = WandbCallback(gradient_save_freq=log_interval,
                                   model_save_path=savedir,
                                   verbose=2)
    agent.learn(total_timesteps=args.steps,
                callback=[eval_callback, wandb_callback],
                log_interval=log_interval)
    agent.save(savedir)


def train_fullinfo(agents, env_test, log_interval, savedir):
    if args.multi_threading:
        fullinfo_agents = FullInfoMultiThreading(agents)
    else:
        fullinfo_agents = PeerFullInfo(agents)
    max_episode_steps = max(args.min_epoch_length,
                            gym.spec(args.env).max_episode_steps)
    n_epochs = args.steps // max_episode_steps
    callbacks = []
    for i in range(len(agents)):
        eval_callback = EvalCallback(env_test,
                                     best_model_save_path=savedir,
                                     log_path=savedir,
                                     eval_freq=log_interval,
                                     n_eval_episodes=args.n_eval_episodes,
                                     deterministic=True, render=False)
        wandb_callback = WandbCallback(gradient_save_freq=log_interval,
                                       model_save_path=savedir,
                                       verbose=2)
        callbacks.append([eval_callback, wandb_callback])

    fullinfo_agents.learn(n_epochs, max_episode_steps, callbacks,
                          log_interval=None)


if __name__ == '__main__':
    parser = add_args()
    args = parser.parse_args()
    time_string = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    unique_dir = f'{time_string}__{wandb.util.generate_id()}'
    experiment_folder = Path.cwd().joinpath("Experiments", args.save_name,
                                            unique_dir)
    print(f'Experiment folder is {experiment_folder}')
    experiment_folder.mkdir(exist_ok=True, parents=True)

    # seed everything
    set_random_seed(args.seed)

    # init wandb
    wandb.tensorboard.patch(root_logdir=str(experiment_folder))
    run = wandb.init(entity='jgu-wandb', config=args.__dict__,
                     project='peer-learning',
                     monitor_gym=True, sync_tensorboard=True,
                     notes=f"Full info with {args.agent_count} agents on "
                           f"the {args.env[:-3]} environment.",
                     dir=str(experiment_folder), mode=args.wandb)
    train_env = make_env()
    test_env = make_env(args.n_eval_episodes)
    if args.track_video:
        def record_video_trigger(x):
            return x % args.eval_interval == 0


        test_env = VecVideoRecorder(test_env, f"videos/{run.id}",
                                    record_video_trigger=record_video_trigger,
                                    video_length=200)
    agents = create_sac_agents(train_env, args.agent_count)
    # setting those values already so the treads don't have to take care of it
    # otherwise only one thread does not throw an exception
    wandb.config.update(agents[0].__dict__, allow_val_change=True)

    if args.agent_count == 1:
        train_single(agents[0], test_env, log_interval=args.eval_interval,
                     savedir=args.save_name)
    else:
        train_fullinfo(agents, test_env, log_interval=args.eval_interval,
                       savedir=args.save_name)
