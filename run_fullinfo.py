import argparse
import datetime

import gym
import pybulletgym  # noqa
import pybullet_envs  # noqa

from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from fullinfo import PeerFullInfo
from fullinfo import FullInfoMultiThreading


# default options for the argument parser
options = {
    # General
    "SAVE_NAME": "full_info",
    "AGENT_COUNT": 4,
    "DEVICE": "auto",
    "ENV": "HalfCheetahBulletEnv-v0",
    # Training
    "STEPS": 3_000_000,
    "EVAL_INTERVAL": 10_000,
    "EVAL_N_RUNS": 10,
    "BUFFER_START_SIZE": 1_000,
    "BUFFER_SIZE": 1_000_000,
    "BATCH_SIZE": 256,
    "MIN_EPOCH_LEN": 1_000,
    "LEARNING_RATE": 3e-4,
    "TAU": 0.005,
    "GAMMA": 0.99,
    "GRAD_STEPS": 1,
    "TRAIN_FREQ": 1,
    # Agents
    "MIX_AGENTS": False,
    "USE_PRIO": False,
    "NET_ARCH": [400, 300],
    # WANDB
    "WANDB": "offline",
}


def add_args():
    # create arg parser
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--save-name", type=str, default=options["SAVE_NAME"])
    parser.add_argument("--device", type=str, default=options["DEVICE"],
                        choices=["cpu", "cuda", "auto"],
                        help="Device to use, either 'cpu', 'cuda' for GPU or "
                             "'auto'.")
    parser.add_argument("--env", type=str, default=options["ENV"],
                        help="OpenAI Gym environment to perform algorithm on.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed in [0, 2 ** 32)")
    parser.add_argument("--track-video", action='store_true')

    parser.add_argument("--agent-count", type=int, help="Number of agents.",
                       default=options["AGENT_COUNT"])
    parser.add_argument("-t", "--multi-threading", action="store_true",
                       help="Run agents parallel in different threads.")
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
    training.add_argument("--batch-size", type=int,
                          default=options["BATCH_SIZE"],
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
    training.add_argument("--train_freq", type=int,
                          default=options["TRAIN_FREQ"])
    # Agents
    agent_parser = parser.add_argument_group("Agent")
    agent_parser.add_argument("--mix-agents", type=bool,
                              default=options["MIX_AGENTS"])
    agent_parser.add_argument("--net-arch", type=list,
                              default=options["NET_ARCH"])
    # WANDB
    parser.add_argument("--wandb", type=str, default=options["WANDB"],
                        choices=["online", "offline", "disabled"])
    return parser


def make_env(seed=0, n_envs=1):
    envs = []
    for s in range(n_envs):
        def env_func():
            env = Monitor(gym.make(args.env))
            env.seed(seed + s)
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
                          log_interval=log_interval)


if __name__ == '__main__':
    parser = add_args()
    args = parser.parse_args()
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
                     dir=str(experiment_folder), mode=args.wandb)
    train_env = make_env(args.seed)
    test_env = make_env(args.seed + 1, args.n_eval_episodes)
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
