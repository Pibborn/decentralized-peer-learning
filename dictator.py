import argparse
import agent_factory
from suggenstionbuffer import SuggestionBuffer
import numpy
import gym
import os
import json
import shutil

import pfrl
from pfrl.agents.ddpg import DDPG
from pfrl import replay_buffers

import pybullet_envs
import pybulletgym
#DEFAULT ARGUMENTS
IS_LOCAL=False#print flag for local experiments
MIN_EPOCH_LEN = 10000
failure_rate = 0.2
options = {
    "FOLLOW_STEPS" : 1,
    "SWITCH_TRAIN" : 1,
    "SWITCH_RATIO" : 1,
    "SAVE_NAME" : "temp",
    "HIDDEN_SIZE" : 256,
    "AGENT_COUNT" : 4,
    "PEER_LEARNING" : 1,
    "USE_AGENT_VALUE" : 0,
    "USE_TRUST" : 1,
    "USE_CRITIC" : 1,
    "T" : 1,
    "T_DECAY" : 0,
    "TRUST_LR" : 0.001,
    "RENDER_TRAIN" : 0,
    "GPU": -1,
    "STEPS": 3_000_000,
    "EVAL_INTERVAL" : 1,
    "EVAL_N_RUNS" : 10,
    "REPLAY_START_SIZE" : 10000,
    "RBUF_CAPACITY" : 10**6,
    "ENV" : "HalfCheetahPyBulletEnv-v0"
}

#ARGUMENT PARSING
parser = argparse.ArgumentParser()

def add_arguments(parser):
    parser.add_argument("--follow-steps",type=int,default=options["FOLLOW_STEPS"])
    parser.add_argument("--hidden-size",type=int,default=options["HIDDEN_SIZE"])
    parser.add_argument("--switch-ratio",type=float,default=options["SWITCH_RATIO"])
    parser.add_argument("--switch-train",type=int,default=options["SWITCH_TRAIN"])
    parser.add_argument("--rbuf-capacity",type=int,default=options["RBUF_CAPACITY"])
    parser.add_argument("--agent-count",type=int,default=options["AGENT_COUNT"])
    parser.add_argument("--use-agent-value",type=int,default=options["USE_AGENT_VALUE"])
    parser.add_argument("--use-trust",type=int,default=options["USE_TRUST"])
    parser.add_argument("--use-critic",type=int,default=options["USE_CRITIC"])
    parser.add_argument("--peer-learning",type=int,default=options["PEER_LEARNING"])
    parser.add_argument("--steps", type=int, default=options["STEPS"], help="Total number of timesteps to train the agent.",
    )
    parser.add_argument("--eval-interval", type=int, default=options["EVAL_INTERVAL"], help="Interval in timesteps between evaluations.")
    parser.add_argument("--eval-n-runs", type=int, default=options["EVAL_N_RUNS"], help="Number of episodes run for each evaluation.",
    )
    parser.add_argument("--save-name",type=str,default=options["SAVE_NAME"])
    parser.add_argument("--trust-lr",type=float,default=options["TRUST_LR"])
    parser.add_argument("--T",type=float,default=options["T"])
    parser.add_argument("--T-decay",type=float,default=options["T_DECAY"])

    parser.add_argument("--env", type=str, default=options["ENV"], help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )

    parser.add_argument("--render-train",type=int,default=options["RENDER_TRAIN"])
    parser.add_argument("--gpu", type=int, default=options["GPU"], help="GPU to use, set to -1 if no GPU.")
    
    parser.add_argument("--seed", type=int, default=1, help="Random seed [0, 2 ** 32)")
   
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

add_arguments(parser)
args = parser.parse_args()
DEVICE = "cuda" if args.gpu == 0 else "cpu"

# ENV SETUP
def make_env(test):
    env = gym.make(args.env)
    # Unwrap TimeLimit wrapper
    assert isinstance(env, gym.wrappers.TimeLimit)
    env = env.env
    # Use different random seeds for train and test envs
    env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
    env.seed(env_seed)
    # Cast observations to float32 because our model uses float32
    env = pfrl.wrappers.CastObservationToFloat32(env)
    # if args.monitor:
    #     env = pfrl.wrappers.Monitor(env, args.outdir)
    if args.render_train and not test:
         env = pfrl.wrappers.Render(env)
         env.render()
    return env

envs = [make_env(test=False),make_env(test=False),make_env(test=False),make_env(test=False)]
env = envs[0]
timestep_limit = env.spec.max_episode_steps
obs_space = env.observation_space
action_space = env.action_space

max_episode_len = env.spec.max_episode_steps
max_epoch_len = max(MIN_EPOCH_LEN,max_episode_len)
n_epochs = args.steps // max_epoch_len

#PATHs
current_dir = os.path.dirname(__file__)
experiment_folder = os.path.join(current_dir,"Experiments",args.save_name)#f"{current_dir}\\Experiments\\{args.save_name}"
ARGS_PATH = os.path.join(experiment_folder,f"{args.save_name}-args.obj")#f"{experiment_folder}\\{args.save_name}_args.obj"
result_folder = os.path.join(experiment_folder,"results.json")#f"{experiment_folder}\\results.json"
save_info_path = os.path.join(experiment_folder,"save_info.json")#f"{experiment_folder}\\results.json"

save_info = False 

assert(args.peer_learning if args.use_trust else True)
assert(args.peer_learning if args.use_critic else True)
assert(args.peer_learning if args.use_agent_value else True)

def create_dir(folder_name):
    try:
        print(folder_name)
        os.mkdir(folder_name)
    except FileExistsError:
        pass
create_dir(experiment_folder)

#CREATE AGENTS
agents = []
sbufs = []
#same buffer for all agents
rbuf = replay_buffers.ReplayBuffer(args.rbuf_capacity)
for i in range(args.agent_count):
    sbuf = SuggestionBuffer(max_epoch_len)
    sbufs.append(sbuf)

    agent = agent_factory.get_sac_agent(obs_space,action_space,args, rbuf)
    agents.append(agent)

#INIT PEER PARAMS AND RESULTS
test_rewards = []
follow_history = [[[0 for _ in range(n_epochs)] for _ in range(len(agents))] for _ in range(len(agents))]
start_epoch = 0

#HELPER FUNCTIONS
def test_agent(agent,env,max_episode_len):
    with agent.eval_mode():
        rewards = []
        for i in range(args.eval_n_runs):
            obs = env.reset()
            R = 0
            t = 0
            while True:
                # Uncomment to watch the behavior in a GUI window
                if args.render_train:
                    env.render()
                action = agent.act(obs)
                obs, r, done, _ = env.step(action)
                R += r
                t += 1

                reset = t == max_episode_len
                agent.observe(obs, r, done, reset)
                if done or reset:
                    break
            rewards.append(R)
            if IS_LOCAL:
                print('evaluation episode:', i, 'R:', R)
        mean = numpy.mean(rewards)
        std = numpy.std(rewards)
        if IS_LOCAL:
            print("mean reward:", mean,"+-",std)
        return mean, std


# boost_single = False
dictator = agents[0]
for i in range(0 + 1, n_epochs + 1):
    for j in range(max_epoch_len):
        for k in range(len(agents)):
            if j==0:
                if k==0:
                    obs = []
                    R = []
                    t = []
                obs.append(envs[k].reset())
                R.append(0)
                t.append(0)
        
            #Uncomment to watch the behavior in a GUI window
            if args.render_train:
                env[k].render()
            
            action = None

            rnd = numpy.random.rand()
            if rnd > failure_rate:
                if k != 0:
                    with dictator.eval_mode():
                        action = dictator.act(obs[k])
                else:
                    action = dictator.act(obs[k])
            else:
                action = numpy.zeros(obs[k].shape)

            previous_obs = obs[k]
            obs[k], reward, done, info = envs[k].step(action)
            
            R[k] += reward
            t[k] += 1
            reset = t[k] == max_episode_len
        
            if k == 0 and rnd > failure_rate:
                dictator.observe(obs[k], reward, done, reset)
            else:
                dictator.replay_buffer.append(previous_obs, action, reward, obs[k])

            if done or reset:
                obs[k] = envs[k].reset()
                R[k] = 0  # return (sum of rewards)
                t[k] = 0  # time step
        
       
    if i % args.eval_interval == 0:

        dictator.save(os.path.join(experiment_folder,'dictatorbackup'))
        dictator.replay_buffer.save(os.path.join(experiment_folder,'dictatorbackup','buffer'))

        save_info = False
        with open(save_info_path, "w") as file:
            json.dump(save_info,file)

        dictator.save(os.path.join(experiment_folder,'dictator'))
        dictator.replay_buffer.save(os.path.join(experiment_folder,'dictator','buffer'))

        save_info = True
        with open(save_info_path, "w") as file:
            json.dump(save_info,file)
        
        
        reward, std = test_agent(dictator,env,max_episode_len)
        test_rewards.append(reward)

        with open(result_folder, "w") as file:
            json.dump({
                "epoch_count" : i,
                "follow_history" : follow_history,
                "rewards" : test_rewards
            },file)



for k in range(args.agent_count):
    if save_info:
        shutil.rmtree(os.path.join(experiment_folder,'dictatorbackup'), ignore_errors=False, onerror=None)
    else:
        shutil.rmtree(os.path.join(experiment_folder,'dictator'), ignore_errors=False, onerror=None)

        