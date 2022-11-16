import argparse
import datetime

import gym
import pybulletgym  # noqa
import pybullet_envs  # noqa

from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed

import wandb
from wandb.integration.sb3 import WandbCallback

from fullinfo import PeerFullInfo
from fullinfo import FullInfoMultiThreading

from utils import add_default_values_to_parser, log_reward_avg_in_wandb,\
    add_default_values_to_train_parser, new_random_seed, make_env, Controller_Arguments


def add_args():
    # create arg parser
    parser = argparse.ArgumentParser(description="Full Information")
    # General
    parser.add_argument("--save-name", type=str, default="full_info")

    parser = add_default_values_to_parser(parser)

    # Training
    training = parser.add_argument_group("Training")
    add_default_values_to_train_parser(training)

    parser.add_argument("--track-video", action="store_true")

    parser.add_argument("-t", "--multi-threading", action="store_true",
                        help="Run agents parallel in different threads.")

    return parser


def create_sac_agents(env, num_agents):
    agent_list = []
    for i in range(num_agents):
        agent = SAC(policy="MlpPolicy", env=env, verbose=1,
                    policy_kwargs=dict(log_std_init=-3,
                                       net_arch=args.net_arch),
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    ent_coef="auto", gamma=args.gamma, tau=args.tau,
                    train_freq=args.train_freq,
                    gradient_steps=args.gradient_steps,
                    learning_starts=args.buffer_start_size, use_sde=True,
                    learning_rate=CA.argument_for_every_agent(args.learning_rate,i),
                    tensorboard_log=str_folder,
                    seed=new_random_seed(),
                    device=args.device)
        agent_list.append(agent)
    return agent_list


def train_single(agent, env_test, log_interval, save_dir):
    eval_callback = EvalCallback(env_test,
                                 best_model_save_path=save_dir,
                                 log_path=save_dir,
                                 eval_freq=log_interval,
                                 deterministic=True, render=False)
    wandb_callback = WandbCallback(gradient_save_freq=log_interval,
                                   verbose=2)
    agent.learn(total_timesteps=args.steps,
                callback=[eval_callback, wandb_callback],
                log_interval=None)

    return [[eval_callback]]


def train_full_info(agents, env_test, log_interval, save_dir):
    if args.multi_threading:
        full_info_agents = FullInfoMultiThreading(agents)
    else:
        full_info_agents = PeerFullInfo(agents)
    max_episode_steps = max(args.min_epoch_length,
                            gym.spec(args.env).max_episode_steps)
    n_epochs = args.steps // max_episode_steps
    callbacks = []
    for i in range(len(agents)):
        eval_callback = EvalCallback(env_test,
                                     best_model_save_path=save_dir,
                                     log_path=save_dir,
                                     eval_freq=log_interval,
                                     n_eval_episodes=args.n_eval_episodes,
                                     deterministic=True, render=False)
        wandb_callback = WandbCallback(gradient_save_freq=log_interval,
                                       verbose=2)
        callbacks.append([wandb_callback, eval_callback])

    full_info_agents.learn(n_epochs, max_episode_steps, callbacks,
                           log_interval=None)

    return callbacks


if __name__ == '__main__':
    # parse args
    arg_parser = add_args()
    args = arg_parser.parse_args()
    CA = Controller_Arguments(args.agent_count)

    # create results/experiments folder
    time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    unique_dir = f"{time_string}__{args.job_id}"
    experiment_folder = Path.cwd().joinpath("Experiments", args.save_name,
                                            unique_dir)
    experiment_folder.mkdir(exist_ok=True, parents=True)
    str_folder = str(experiment_folder)
    print("Experiment folder is", str_folder)

    # seed everything
    set_random_seed(args.seed)

    # init wandb
    wandb.tensorboard.patch(root_logdir=str_folder)
    run = wandb.init(entity="jgu-wandb", config=args.__dict__,
                     project="peer-learning",
                     monitor_gym=True, sync_tensorboard=True,
                     notes=f"Full info with {args.agent_count} agents on "
                           f"the {args.env[:-3]} environment.",
                     dir=str_folder, mode=args.wandb)

    train_env = make_env(args.env, **args.env_args)
    test_env = make_env(args.env, args.n_eval_episodes, **args.env_args)
    if args.track_video:
        def record_video_trigger(x):
            return x % args.eval_interval == 0


        test_env = VecVideoRecorder(test_env, f"videos/{run.id}",
                                    record_video_trigger=record_video_trigger,
                                    video_length=200)
    agents = create_sac_agents(train_env, args.agent_count)
    if args.multi_threading:
        # setting those values already so the threads don't have to take
        # care of it otherwise only one thread does not throw an exception
        wandb.config.update(agents[0].__dict__, allow_val_change=True)

    if args.agent_count == 1:
        callbacks = train_single(agents[0], test_env,
                                 log_interval=args.eval_interval,
                                 save_dir=args.save_name)
    else:
        callbacks = train_full_info(agents, test_env,
                                    log_interval=args.eval_interval,
                                    save_dir=args.save_name)

    log_reward_avg_in_wandb(callbacks)
