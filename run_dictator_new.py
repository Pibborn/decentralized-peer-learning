import argparse
import datetime

import gym
import pybulletgym  # noqa
import pybullet_envs  # noqa

from pathlib import Path

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

# rename for convenience
from peer import PeerGroup as Dictator
from new_dictator import make_dictator_class

# default options for the argument parser
options = {
    # General
    "SAVE_NAME": "dictator",
    "AGENT_COUNT": 4,
    "DEVICE": "auto",
    "ENV": "HalfCheetahBulletEnv-v0",
    # Training
    "STEPS": 3_000_000,
    "EVAL_INTERVAL": 10_000,
    "EVAL_N_RUNS": 10,
    "BUFFER_START_SIZE": 1_000,
    "BUFFER_SIZE": 1_000_000,
    "BATCH_SIZE": 100,
    "MIN_EPOCH_LEN": 10_000,
    "LEARNING_RATE": 3e-4,
    "TAU": 0.005,
    "GAMMA": 0.99,
    "GRAD_STEPS": 1,
    "TRAIN_FREQ": 1,
    # Agents
    "MIX_AGENTS": False,
    "NET_ARCH": [400, 300],
    # Dictator
    "T": 1,
    "T_DECAY": 0,
    # WANDB
    "WANDB": "offline",
}


def add_args():
    # create arg parser
    parser = argparse.ArgumentParser(description="Dictator.")
    # General
    parser.add_argument("--save-name", type=str, default=options["SAVE_NAME"])
    parser.add_argument("--agent-count", type=int, help="Number of agents.",
                        default=options["AGENT_COUNT"])
    parser.add_argument("--device", type=str, default=options["DEVICE"],
                        choices=["cpu", "cuda", "auto"],
                        help="Device to use, either 'cpu', 'cuda' for GPU or "
                             "'auto'.")
    parser.add_argument("--env", type=str, default=options["ENV"],
                        help="OpenAI Gym environment to perform algorithm on.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed in [0, 2 ** 32)")
    # Training
    training = parser.add_argument_group("Training")
    training.add_argument("--steps", type=int, default=options["STEPS"],
                          help="Total number of time steps to train the agent.")
    training.add_argument("--eval-interval", type=int,
                          default=options["EVAL_INTERVAL"],
                          help="Interval in time steps between evaluations.")
    training.add_argument("--n-eval-episodes", type=int,
                          default=options["EVAL_N_RUNS"],
                          help="Number of episodes for each evaluation.")
    training.add_argument("--buffer-size", type=int,
                          default=options["BUFFER_SIZE"])
    training.add_argument("--buffer-start-size", type=int,
                          default=options["BUFFER_START_SIZE"],
                          help="Minimum replay buffer size before performing "
                               "gradient updates.")
    training.add_argument("--batch-size", type=int, default=options["BATCH_SIZE"],
                          help="Minibatch size")
    training.add_argument("--min-epoch-length", type=int,
                          default=options["MIN_EPOCH_LEN"],
                          help="Minimal length of a training epoch.")
    training.add_argument("--learning_rate", type=float,
                          default=options["LEARNING_RATE"])
    training.add_argument("--tau", type=float, default=options["TAU"])
    training.add_argument("--gamma", type=float, default=options["GAMMA"])
    training.add_argument("--gradient_steps", type=int,
                          default=options["GRAD_STEPS"])
    training.add_argument("--train_freq", type=int, default=options["TRAIN_FREQ"])
    # Agents
    agent_parser = parser.add_argument_group("Agent")
    agent_parser.add_argument("--mix-agents", type=bool,
                              default=options["MIX_AGENTS"])
    agent_parser.add_argument("--net-arch", type=list,
                              default=options["NET_ARCH"])
    # Dictator
    dictator = parser.add_argument_group("Dictator")
    dictator.add_argument("--T", type=float, default=options["T"])
    dictator.add_argument("--T-decay", type=float, default=options["T_DECAY"])
    # WANDB
    parser.add_argument("--wandb", type=str, default=options["WANDB"],
                        choices=["online", "offline", "disabled"])
    return parser


# environment function
def make_env(seed=0):
    env = gym.make(args.env)
    env.seed(seed)
    env = Monitor(env)
    return DummyVecEnv([lambda: env])


if __name__ == '__main__':
    # parse args
    parser = add_args()
    args = parser.parse_args()

    # create results/experiments folder
    time_string = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    unique_dir = f'{time_string}__{wandb.util.generate_id()}'
    experiment_folder = Path.cwd().joinpath("Experiments", args.save_name,
                                            unique_dir)
    experiment_folder.mkdir(exist_ok=True, parents=True)

    # init wandb
    wandb.tensorboard.patch(root_logdir=str(experiment_folder))
    run = wandb.init(entity='jgu-wandb', config=args.__dict__,
                     project='peer-learning',
                     monitor_gym=True, sync_tensorboard=True,
                     notes=f"Dictator with {args.agent_count} agents on "
                           f"the {args.env[:-3]} environment.",
                     dir=str(experiment_folder), mode=args.wandb)

    # initialize dictator
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
                     sample_actions=True, greedy_suggestions=True)

    # create Dictator classes
    SACSub = make_dictator_class(SAC)
    TD3Sub = make_dictator_class(TD3)

    # create subs
    subs = []
    callbacks = []
    for i in range(args.agent_count):
        if args.mix_agents and i % 2 != 0:
            subs.append(TD3Sub(**peer_args))
        else:
            subs.append(SACSub(**peer_args))

        # every agent gets its own callbacks
        callbacks.append([EvalCallback(eval_env=make_env(args.seed),
                                       best_model_save_path=str(experiment_folder),
                                       log_path=str(experiment_folder),
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
                   eval_log_path=str(experiment_folder),
                   max_epoch_len=max_episode_steps)
