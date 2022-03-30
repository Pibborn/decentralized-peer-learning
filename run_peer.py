import argparse
import gym
import pybulletgym  # noqa
import pybullet_envs  # noqa

from pathlib import Path

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from peer import PeerGroup, make_peer_class

# default options for the argument parser
options = {
    "FOLLOW_STEPS": 1,
    "MIX_AGENTS": False,
    "SWITCH_RATIO": 1,
    "SAVE_NAME": "Peers",
    "LOAD": False,
    "NET_ARCH": [400, 300],
    "AGENT_COUNT": 4,
    "PEER_LEARNING": True,
    "USE_AGENT_VALUE": True,
    "USE_TRUST": True,
    "USE_TRUST_BUFFER": True,
    "USE_CRITIC": True,
    "T": 1,
    "T_DECAY": 0,
    "TRUST_LR": 0.001,
    "DEVICE": "auto",
    "STEPS": 3_000_000,
    "EVAL_INTERVAL": 10_000,
    "EVAL_N_RUNS": 10,
    "BUFFER_START_SIZE": 1_000,
    "BUFFER_SIZE": 1_000_000,
    "ENV": "HalfCheetahBulletEnv-v0",
    "BATCH_SIZE": 100,
    "MIN_EPOCH_LEN": 10_000,
    "WANDB": "disabled",
}

# create arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--follow-steps", type=int,
                    default=options["FOLLOW_STEPS"])
parser.add_argument("--net-arch", type=list,
                    default=options["NET_ARCH"])
parser.add_argument("--mix-agents", type=bool,
                    default=options["MIX_AGENTS"])
parser.add_argument("--switch-ratio", type=float,
                    default=options["SWITCH_RATIO"],
                    help="Ratio of peer learning episodes to solo episodes; "
                         "0 means only peer learning episodes.")
parser.add_argument("--buffer-size", type=int,
                    default=options["BUFFER_SIZE"])
parser.add_argument("--agent-count", type=int,
                    default=options["AGENT_COUNT"])
parser.add_argument("--use-agent-value", type=bool,
                    default=options["USE_AGENT_VALUE"])
parser.add_argument("--use-trust", type=bool, default=options["USE_TRUST"])
parser.add_argument("--use-trust-buffer", type=bool,
                    default=options["USE_TRUST_BUFFER"])
parser.add_argument("--use-critic", type=bool,
                    default=options["USE_CRITIC"])
parser.add_argument("--peer-learning", type=bool,
                    default=options["PEER_LEARNING"])
parser.add_argument("--steps", type=int, default=options["STEPS"],
                    help="Total number of time steps to train the agent.")
parser.add_argument("--eval-interval", type=int,
                    default=options["EVAL_INTERVAL"],
                    help="Interval in time steps between evaluations.")
parser.add_argument("--n-eval-episodes", type=int,
                    default=options["EVAL_N_RUNS"],
                    help="Number of episodes for each evaluation.")
parser.add_argument("--trust-lr", type=float, default=options["TRUST_LR"])
parser.add_argument("--save-name", type=str, default=options["SAVE_NAME"])
parser.add_argument("--T", type=float, default=options["T"])
parser.add_argument("--T-decay", type=float, default=options["T_DECAY"])
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--train_freq", type=int, default=1)
parser.add_argument("--env", type=str, default=options["ENV"],
                    help="OpenAI Gym environment to perform algorithm on.")
parser.add_argument("--device", type=str, default=options["DEVICE"],
                    choices=["cpu", "cuda", "auto"],
                    help="Device to use, either 'cpu', 'cuda' for GPU or "
                         "'auto'.")
parser.add_argument("--load", type=bool, default=options["LOAD"])
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed in [0, 2 ** 32)")
parser.add_argument("--buffer-start-size", type=int,
                    default=options["BUFFER_START_SIZE"],
                    help="Minimum replay buffer size before performing "
                         "gradient updates.")
parser.add_argument("--batch-size", type=int, default=options["BATCH_SIZE"],
                    help="Minibatch size")
parser.add_argument("--min-epoch-length", type=int,
                    default=options["MIN_EPOCH_LEN"],
                    help="Minimal length of a training epoch.")
parser.add_argument("--wandb", type=str, default=options["WANDB"])

# parse args
args = parser.parse_args()

# assert if any peer learning strategy is chosen peer learning must be True
option_on = (args.use_trust or args.use_critic or args.use_agent_value)
assert (option_on and args.peer_learning) or not option_on

# create results/experiments folder
experiment_folder = Path.cwd().joinpath("Experiments", args.save_name,
                                        wandb.util.generate_id())
experiment_folder.mkdir(exist_ok=True, parents=True)

# init wandb
wandb.tensorboard.patch(root_logdir=str(experiment_folder))
run = wandb.init(entity='jgu-wandb', config=args.__dict__,
                 project='peer-learning',
                 monitor_gym=True, sync_tensorboard=True,
                 notes=f"Peer Learning with {args.agent_count} agents on "
                       f"the {args.env[:-3]} environment.",
                 dir=str(experiment_folder), mode=args.wandb)


# environment function
def make_env(seed=0):
    env = gym.make(args.env)
    env.seed(seed)
    env = Monitor(env)
    return env


# initialize peer group
algo_args = dict(policy="MlpPolicy", verbose=1,
                 policy_kwargs=dict(log_std_init=-3, net_arch=args.net_arch),
                 buffer_size=args.buffer_size,
                 batch_size=args.batch_size,
                 ent_coef='auto', gamma=args.gamma, tau=args.tau, train_freq=args.train_freq,
                 learning_starts=args.buffer_start_size, use_sde=True,
                 learning_rate=args.learning_rate, gradient_steps=args.gradient_steps,
                 tensorboard_log=str(experiment_folder),
                 device=args.device)

peer_args = dict(temperature=args.T, temp_decay=args.T_decay,
                 algo_args=algo_args, env_func=make_env,
                 use_trust=args.use_trust, use_critic=args.use_critic,
                 buffer_size=args.buffer_size,
                 follow_steps=args.follow_steps,
                 use_trust_buffer=args.use_trust_buffer,
                 solo_training=not args.peer_learning)

# create Peer classes
SACPeer = make_peer_class(SAC)
TD3Peer = make_peer_class(TD3)

# create peers
peers = []
callbacks = []
for i in range(args.agent_count):
    if args.mix_agents and i % 2 != 0:
        peers.append(TD3Peer(**peer_args))
    else:
        peers.append(SACPeer(**peer_args))

    # every agent gets its own callbacks
    callbacks.append([EvalCallback(eval_env=make_env(args.seed),
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
n_epochs = args.steps // max_episode_steps

# train the peer group
peer_group.learn(n_epochs, callbacks=callbacks,
                 eval_log_path=str(experiment_folder),
                 max_epoch_len=max_episode_steps)
