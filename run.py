import os
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import argparse
import gym
import datetime
import pybulletgym
import pybullet_envs
import wandb
from wandb.integration.sb3 import WandbCallback

options = {
    "FOLLOW_STEPS" : 1,
    "MIX_AGENTS" : 0,
    "SWITCH_TRAIN" : 1,
    "SWITCH_RATIO" : 1,
    "USE_PRIO" : 0,
    "SAVE_NAME" : "temp",
    "LOAD" : 0,
    "HIDDEN_SIZE" : 256,
    "AGENT_COUNT" : 4,
    "PEER_LEARNING" : 1,
    "USE_AGENT_VALUE" : 0,
    "USE_TRUST" : 1,
    "USE_TRUST_BUFFER" : 1,
    "USE_CRITIC" : 1,
    "T" : 1,
    "T_DECAY" : 0,
    "TRUST_LR" : 0.001,
    "RENDER_TRAIN" : 0,
    "GPU": -1,
    "STEPS": 3_000_000,#200*100,#
    "EVAL_INTERVAL" : 1,
    "EVAL_N_RUNS" : 10,
    "REPLAY_START_SIZE" : 10000,#,200,#
    "RBUF_CAPACITY" : 10**6,
    "TEST": False,
    "ENV" : "HalfCheetahBulletEnv-v0",#Pendulum-v0"#""Pendulum-v0"#"Walker2DPyBulletEnv-v0" #"InvertedDoublePendulumPyBulletEnv-v0"#"BipedalWalker-v3"#
    "N_AGENTS": 4,
    "BASESAVELOC": "./agents/",
    "SAVEDIR": datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
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
    parser.add_argument("--track-video", action='store_true')

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


def make_env(seed):
    env = gym.make(args.env)
    # Unwrap TimeLimit wrapper TODO understand why
    # assert isinstance(env, gym.wrappers.TimeLimit)
    # env = env.env
    # Use different random seeds for train and test envs
    env_seed = seed
    env.seed(env_seed)
    #env._max_episode_steps = 1e4 # see https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
    env = Monitor(env)
    if args.render_train:
        env.render()

    def return_env():
        return env
    return return_env


def create_sac_agents(env, num_agents):
    agent_list = []
    for i in range(num_agents):
        # hps taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
        agent = SAC("MlpPolicy", env, policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]), verbose=1,
                    buffer_size=300000, batch_size=256, ent_coef='auto', gamma=0.98, tau=0.02, train_freq=64,
                    learning_starts=10000, use_sde=True, learning_rate=7.3e-4, gradient_steps=64,
                    tensorboard_log='agents/sac')
        agent_list.append(agent)
    return agent_list


def train_single(agent, env_train, env_test, log_interval=1000, savedir='agent'):
    eval_callback = EvalCallback(env_test, best_model_save_path='./agents/',
                                 log_path='./logs/', eval_freq=log_interval,
                                 deterministic=True, render=False)
    wandb_callback = WandbCallback(gradient_save_freq=log_interval,
                                   model_save_path="results/temp",
                                   verbose=2)
    agent.learn(total_timesteps=3e6, callback=[eval_callback, wandb_callback], log_interval=log_interval)
    agent.save(args.basedir + os.sep + 'single_' + savedir)
    rewards = evaluate_policy(agents[0], test_env)
    print(rewards)


if __name__ == '__main__':
    parser = add_args()
    args = parser.parse_args()
    run = wandb.init(entity='jgu-wandb', config=args, project='peer-learning', monitor_gym=True, sync_tensorboard=True)
    #train_env = SubprocVecEnv([make_env(args.env, args.test) for i in range(args.num_agents)])
    #train_env = make_vec_env(args.env, n_envs=args.num_agents, seed=args.seed+1)
    train_env = DummyVecEnv([make_env(args.seed)])
    test_env = DummyVecEnv([make_env(args.seed+1)])
    if args.track_video:
        test_env = VecVideoRecorder(test_env, f"videos/{run.id}",
                                    record_video_trigger=lambda x: x % args.log_interval == 0,
                                    video_length=200)
    check_args(args)
    agents = create_sac_agents(train_env, args.num_agents)
    train_single(agents[0], train_env, test_env, log_interval=args.log_interval, savedir=args.savedir)

