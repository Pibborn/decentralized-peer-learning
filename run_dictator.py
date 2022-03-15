import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import argparse
import gym
from gym.spaces import Box
import datetime
import pybulletgym
import pybullet_envs
import wandb
from wandb.integration.sb3 import WandbCallback

options = {
    "FOLLOW_STEPS": 1,
    "MIX_AGENTS": 0,
    "SWITCH_TRAIN": 1,
    "SWITCH_RATIO": 1,
    "USE_PRIO": 0,
    "SAVE_NAME": "temp",
    "LOAD": 0,
    "HIDDEN_SIZE": 256,
    "AGENT_COUNT": 4,  # is that ever used?
    "PEER_LEARNING": 1,
    "USE_AGENT_VALUE": 0,
    "USE_TRUST": 1,
    "USE_TRUST_BUFFER": 1,
    "USE_CRITIC": 1,
    "T": 1,
    "T_DECAY": 0,
    "TRUST_LR": 0.001,
    "RENDER_TRAIN": 0,
    "GPU": -1,
    "STEPS": 3_000_000,
    "EVAL_INTERVAL": 1,
    "EVAL_N_RUNS": 10,
    "REPLAY_START_SIZE": 10_000,
    "RBUF_CAPACITY": 300_000,
    "TEST": False,
    "ENV": "HalfCheetahBulletEnv-v0",
    "N_AGENTS": 3,
    "BASESAVELOC": "./agents/",
    "SAVEDIR": datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'),
    "LOGINTERVAL": 10_000,
    "BATCH_SIZE": 100,
}


def check_args(args):
    assert(args.peer_learning if args.use_trust else True)
    assert(args.peer_learning if args.use_critic else True)
    assert(args.peer_learning if args.use_agent_value else True)

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--follow-steps", type=int, default=options["FOLLOW_STEPS"])
    parser.add_argument("--hidden-size", type=int, default=options["HIDDEN_SIZE"])
    parser.add_argument("--mix-agents", type=int, default=options["MIX_AGENTS"])
    parser.add_argument("--switch-ratio", type=float, default=options["SWITCH_RATIO"])
    parser.add_argument("--switch-train", type=int, default=options["SWITCH_TRAIN"])
    parser.add_argument("--use-prio", type=int, default=options["USE_PRIO"])
    parser.add_argument("--rbuf-capacity", type=int, default=options["RBUF_CAPACITY"])
    parser.add_argument("--agent-count", type=int, default=options["AGENT_COUNT"])
    parser.add_argument("--use-agent-value", type=int, default=options["USE_AGENT_VALUE"])
    parser.add_argument("--use-trust", type=int, default=options["USE_TRUST"])
    parser.add_argument("--use-trust-buffer", type=int, default=options["USE_TRUST_BUFFER"])
    parser.add_argument("--use-critic", type=int, default=options["USE_CRITIC"])
    parser.add_argument("--peer-learning", type=int, default=options["PEER_LEARNING"])
    parser.add_argument("--steps", type=int, default=options["STEPS"],
                        help="Total number of timesteps to train the agent.",
                        )
    parser.add_argument("--eval-interval", type=int, default=options["EVAL_INTERVAL"],
                        help="Interval in timesteps between evaluations.")
    parser.add_argument("--eval-n-runs", type=int, default=options["EVAL_N_RUNS"],
                        help="Number of episodes run for each evaluation.",
                        )
    parser.add_argument("--save-name", type=str, default=options["SAVE_NAME"])
    parser.add_argument("--trust-lr", type=float, default=options["TRUST_LR"])
    parser.add_argument("--T", type=float, default=options["T"])
    parser.add_argument("--T-decay", type=float, default=options["T_DECAY"])

    parser.add_argument("--env", type=str, default=options["ENV"],
                        help="OpenAI Gym MuJoCo env to perform algorithm on.",
                        )

    parser.add_argument("--render-train", type=int, default=options["RENDER_TRAIN"])
    parser.add_argument("--gpu", type=int, default=options["GPU"], help="GPU to use, set to -1 if no GPU.")
    parser.add_argument("--load", type=int, default=options["LOAD"])
    parser.add_argument("--seed", type=int, default=1, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--test", action='store_true', default=options['TEST'])
    parser.add_argument("--num-agents", type=int, default=options["N_AGENTS"])
    parser.add_argument("--basedir", type=str, default=options["BASESAVELOC"])
    parser.add_argument("--savedir", type=str, default=options["SAVEDIR"])
    parser.add_argument("--log-interval", type=int, default=options["LOGINTERVAL"])

    parser.add_argument("--policy-output-scale",
                        type=float,
                        default=1.0,
                        help="Weight initialization scale of policy output.")

    parser.add_argument("--replay-start-size",
                        type=int,
                        default=options["REPLAY_START_SIZE"],
                        help="Minimum replay buffer size before performing "
                             "gradient updates.")

    parser.add_argument("--batch-size", type=int,
                        default=options["BATCH_SIZE"],
                        help="Minibatch size")
    return parser


def make_env(seed):
    env = gym.make(args.env)
    # Unwrap TimeLimit wrapper TODO understand why
    # assert isinstance(env, gym.wrappers.TimeLimit)
    # env = env.env
    # Use different random seeds for train and test envs
    env.seed(seed)
    # env._max_episode_steps = 1e6 # see https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
    env = Monitor(env)
    if args.render_train:
        env.render()

    def return_env():
        return env
    return return_env


class SumReward(gym.Env):
    def __init__(self, env):
        super(SumReward, self).__init__()
        self.env = env

        a_space = env.action_space
        o_space = env.observation_space
        self.action_space = Box(np.tile(a_space.low,
                                        env.num_envs),
                                np.tile(a_space.high,
                                        env.num_envs))

        self.observation_space = Box(np.tile(o_space.low,
                                             env.num_envs),
                                     np.tile(o_space.high,
                                             env.num_envs))

    def reset(self):
        return np.reshape(self.env.reset(), -1)

    def render(self, mode="human"):
        return self.env.render()

    def step(self, action: np.ndarray):
        o, r, done, infos = self.env.step(np.reshape(action,
                                                     (self.env.num_envs, -1)))
        return np.reshape(o, -1), r.mean(), np.any(done), infos[0]


def create_sac_agents(env, num_agents):
    agent_list = []
    for i in range(num_agents):
        # hps taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
        agent = SAC("MlpPolicy", env, verbose=1,
                    policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
                    buffer_size=args.rbuf_capacity,
                    batch_size=args.batch_size,
                    ent_coef='auto',
                    gamma=0.98, tau=0.02, train_freq=8,
                    learning_starts=args.replay_start_size, use_sde=True,
                    learning_rate=7.3e-4,
                    gradient_steps=8, tensorboard_log='agents/dictator',
                    device='cuda')
        agent_list.append(agent)
    return agent_list


def train_dictator(agent, env_train, env_test, log_interval=1000,
                   savedir='agent'):
    eval_callback = EvalCallback(env_test, best_model_save_path='./agents/',
                                 log_path='./logs/', eval_freq=log_interval,
                                 deterministic=True, render=False)
    wandb_callback = WandbCallback(gradient_save_freq=log_interval,
                                   model_save_path="results/temp",
                                   verbose=2)
    agent.learn(total_timesteps=args.steps,
                # callback=[eval_callback],
                callback=[eval_callback, wandb_callback],
                log_interval=log_interval)
    agent.save(args.basedir + os.sep + 'dictator_' + savedir)
    rewards = evaluate_policy(agents[0], env_test)
    print(rewards)


if __name__ == '__main__':
    parser = add_args()
    args = parser.parse_args()

    wandb.init(entity='jgu-wandb', config=vars(args),
               project='peer-learning',
               monitor_gym=True,  sync_tensorboard=True)
    train_env = SumReward(SubprocVecEnv(
        [make_env(args.seed) for i in range(args.num_agents)]))
    test_env = SumReward(SubprocVecEnv(
        [make_env(args.seed + 1) for i in range( args.num_agents)]))

    check_args(args)
    agents = create_sac_agents(train_env, 1)
    train_dictator(agents[0], train_env, test_env,
                   log_interval=args.log_interval, savedir=args.savedir)

