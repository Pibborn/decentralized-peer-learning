import argparse
import datetime

import gym
import pybulletgym  # noqa: F401
import pybullet_envs  # noqa: F401

from pathlib import Path

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

import wandb
from wandb.integration.sb3 import WandbCallback

# rename for convenience
from peer import PeerGroup as Dictator
from new_dictator import make_dictator_class, make_weighted_dictator_class

from utils import add_default_values_to_parser, log_reward_avg_in_wandb, \
    add_default_values_to_train_parser, new_random_seed, make_env,\
    ControllerArguments, str2bool


def add_args():
    # create arg parser
    parser = argparse.ArgumentParser(description="Dictator.")
    # General
    parser.add_argument("--save-name", type=str, default="dictator")
    parser = add_default_values_to_parser(parser)

    # Training
    training = parser.add_argument_group("Training")
    add_default_values_to_train_parser(training)

    # Dictator
    dictator_group = parser.add_argument_group("Dictator")
    dictator_group.add_argument("--T", type=float, default=1)
    dictator_group.add_argument("--T-decay", type=float, default=0)
    dictator_group.add_argument("--weighted-dictator", type=str2bool,
                                nargs="?", const=True, default=False)

    return parser


if __name__ == '__main__':
    # parse args
    arg_parser = add_args()
    args = arg_parser.parse_args()
    CA = ControllerArguments(args.agent_count)

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
                     notes=f"Dictator with {args.agent_count} agents on "
                           f"the {args.env[:-3]} environment.",
                     dir=str_folder, mode=args.wandb)

    # initialize dictator
    peer_args = []
    for i in range(args.agent_count):
        algo_args = dict(policy="MlpPolicy", verbose=1,
                         policy_kwargs=dict(log_std_init=-3,
                                            net_arch=args.net_arch),
                         buffer_size=args.buffer_size,
                         batch_size=args.batch_size,
                         ent_coef="auto", gamma=args.gamma, tau=args.tau,
                         train_freq=args.train_freq,
                         gradient_steps=args.gradient_steps,
                         learning_starts=args.buffer_start_size, use_sde=True,
                         learning_rate=CA.argument_for_every_agent(
                             args.learning_rate, i),
                         tensorboard_log=str_folder,
                         device=args.device)
        peer_args.append(
            dict(temperature=args.T, temp_decay=args.T_decay,
                 algo_args=algo_args, env=args.env,
                 sample_actions=True, peers_sample_with_noise=False))

    # create Dictator classes
    if args.weighted_dictator:
        SACSub = make_weighted_dictator_class(SAC)
        TD3Sub = make_weighted_dictator_class(TD3)
    else:
        SACSub = make_dictator_class(SAC)
        TD3Sub = make_dictator_class(TD3)

    # create subs
    subs = []
    callbacks = []
    for i in range(args.agent_count):
        args_for_agent = peer_args[i]
        if args.mix_agents and i % 2 != 0:
            subs.append(TD3Sub(**args_for_agent, seed=new_random_seed()))
        else:
            subs.append(SACSub(**args_for_agent, seed=new_random_seed()))

        # every agent gets its own callbacks
        callbacks.append([EvalCallback(eval_env=make_env(args.env),
                                       best_model_save_path=str_folder,
                                       log_path=str_folder,
                                       eval_freq=args.eval_interval,
                                       n_eval_episodes=args.n_eval_episodes),
                          WandbCallback(gradient_save_freq=args.eval_interval,
                                        verbose=2)])

    # create dictator from submissive agents
    dictator = Dictator(subs)

    # calculate number of epochs based on episode length
    max_episode_steps = max(args.min_epoch_length,
                            gym.spec(args.env).max_episode_steps)

    n_epochs = args.steps // max_episode_steps

    # train the dictator
    dictator.learn(n_epochs, callbacks=callbacks,
                   eval_log_path=str_folder,
                   max_epoch_len=max_episode_steps)

    log_reward_avg_in_wandb(callbacks)
