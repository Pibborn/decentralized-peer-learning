import argparse
import wandb
import gym

import numpy as np

from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env(env_str, n_envs=1):
    envs = []
    for _ in range(n_envs):
        def env_func():
            env = Monitor(gym.make(env_str))
            env.seed(new_random_seed())
            return env

        envs.append(env_func)
    return DummyVecEnv(envs)


def new_random_seed():
    return np.random.randint(np.iinfo(np.int32).max)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2func(v):
        function  = eval(v)
        return function
class Controller_Arguments():
    def __init__(self, number_agents):
        self.number_agents = number_agents

    def argument_for_every_agent(self,input, i):
        if len(input) == 1:
            return input[0]
        elif len(input) == self.number_agents:
            return input[i]
        else:
            raise AssertionError(f'number of arguments {len(input)} has'
                                 f' to be 1 or == number of agents ({self.number_agents})')


def add_default_values_to_parser(parser):
    parser.add_argument("--job_id", type=str,
                        default=wandb.util.generate_id())
    parser.add_argument("--agent-count", type=int, help="Number of agents.",
                        default=4)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "auto"],
                        help="Device to use, either 'cpu', 'cuda' for GPU or "
                             "'auto'.")
    parser.add_argument("--env", type=str, default="HalfCheetahBulletEnv-v0",
                        help="OpenAI Gym environment to perform algorithm on.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed in [0, 2 ** 32)")
    parser.add_argument("--wandb", type=str, default='offline',
                        choices=["online", "offline", "disabled"])
    # Agents
    agent_parser = parser.add_argument_group("Agent")
    agent_parser.add_argument("--mix-agents", type=str, nargs='*',
                              default = "SAC")

    agent_parser.add_argument("--net-arch", type=int, nargs='*',
                              default= [400, 300])

    return parser


def add_default_values_to_train_parser(training_parser):
    training_parser.add_argument("--steps", type=int, default=3_000_000,
                                 help="Total number of time steps to train "
                                      "the agent.")
    training_parser.add_argument("--eval-interval", type=int,
                                 default=10_000,
                                 help="Interval in time steps between "
                                      "evaluations.")
    training_parser.add_argument("--n-eval-episodes", type=int,
                                 default=10,
                                 help="Number of episodes for each "
                                      "evaluation.")
    training_parser.add_argument("--buffer-size", type=int,
                                 default=1_000_000)
    training_parser.add_argument("--buffer-start-size", type=int,
                                 default=1_000,
                                 help="Minimum replay buffer size before "
                                      "performing gradient updates.")
    training_parser.add_argument("--batch-size", type=int,
                                 default=100,
                                 help="Minibatch size")
    training_parser.add_argument("--min-epoch-length", type=int,
                                 default=10_000,
                                 help="Minimal length of a training_parser "
                                      "epoch.")
    training_parser.add_argument("--learning_rate", type=str2func, nargs='*',
                                 default=3e-4)
    training_parser.add_argument("--tau", type=float, default=0.005)
    training_parser.add_argument("--gamma", type=float, default=0.99)
    training_parser.add_argument("--gradient_steps", type=int,
                                 default=1)
    training_parser.add_argument("--train_freq", type=int,
                                 default=1)
    return training_parser




def log_reward_avg_in_wandb(callbacks):
    results = []
    for callback in callbacks:
        eval_callback = callback[0]
        result = eval_callback.evaluations_results
        results.append(np.mean(result))
    wandb.log({'reward_avg': np.mean(results)})
