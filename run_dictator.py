import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
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
    "HIDDEN_SIZE": 20,
    "AGENT_COUNT": 5,
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
    "STEPS": 500_000,
    "EVAL_INTERVAL": 1,
    "EVAL_N_RUNS": 20,
    "REPLAY_START_SIZE" : 10000,#,200,#
    "RBUF_CAPACITY": 10**6,
    "TEST": False,
    "ENV": "Pendulum-v0",#"#HalfCheetahPyBulletEnv-v0""Pendulum-v0"#"Walker2DPyBulletEnv-v0" #"InvertedDoublePendulumPyBulletEnv-v0"#"BipedalWalker-v3"#
    "N_AGENTS": 6,
    "BASESAVELOC": "./agents/",
    "SAVEDIR": datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'),
    "LOGINTERVAL": 1000
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

    parser.add_argument(
        "--policy-output-scale",
        type=float,
        default=1.0,
        help="Weight initialization scale of policy output.",
    )

    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=options["REPLAY_START_SIZE"],
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Minibatch size")
    return parser


def make_env(env_str, is_test):
    env = gym.make(env_str)
    # Unwrap TimeLimit wrapper TODO understand why
    # assert isinstance(env, gym.wrappers.TimeLimit)
    # env = env.env
    # Use different random seeds for train and test envs
    env_seed = args.seed if is_test else args.seed - 1
    env.seed(env_seed)
    # env._max_episode_steps = 1e6 # see https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
    # Cast observations to float32 because our model uses float32
    # if args.monitor:
    #     env = pfrl.wrappers.Monitor(env, args.outdir)
    if args.render_train:
        env.render()
    return env


def env_func(env_str, is_test):
    return lambda: make_env(env_str, is_test)


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
        o, r, done, infos = self.env.step(np.reshape(action, (-1, 1)))
        return np.reshape(o, -1), r.mean(), np.any(done), infos[0]


def create_sac_agents(env, num_agents):
    agent_list = []
    for i in range(num_agents):
        # hps taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
        agent = SAC("MlpPolicy", env, policy_kwargs=dict(log_std_init=-3,
                                                         net_arch=[100, 100]),
                    verbose=1,
                    buffer_size=30000, batch_size=24, ent_coef='auto',
                    gamma=0.98, tau=0.02, train_freq=8,
                    learning_starts=1000, use_sde=True, learning_rate=4e-4,
                    gradient_steps=8)
        agent_list.append(agent)
    return agent_list


def train_dictator(agent, env_train, env_test, log_interval=1000,
                savedir='agent'):
    eval_callback = EvalCallback(env_test, best_model_save_path='./agents/',
                                 log_path='./logs/', eval_freq=1000,
                                 deterministic=True, render=False)
    wandb_callback = WandbCallback(gradient_save_freq=1000,
                                   model_save_path="results/temp",
                                   verbose=0)
    agent.learn(total_timesteps=args.steps,
                callback=[eval_callback, wandb_callback],
                log_interval=log_interval)
    agent.save(args.basedir + os.sep + 'single_' + savedir)
    rewards = evaluate_policy(agents[0], test_env)
    print(rewards)


if __name__ == '__main__':
    parser = add_args()
    args = parser.parse_args()

    wandb.init(entity='jgu-wandb', config=args, project='peer-learning', monitor_gym=True)
    train_env = SumReward(SubprocVecEnv([env_func(args.env, args.test) for i in
                               range(args.num_agents)]))
    test_env = SumReward(SubprocVecEnv([env_func(args.env, False) for i in range(
        args.num_agents)]))

    check_args(args)
    agents = create_sac_agents(train_env, 1)
    train_dictator(agents[0], train_env, test_env,
                   log_interval=args.log_interval, savedir=args.savedir)

