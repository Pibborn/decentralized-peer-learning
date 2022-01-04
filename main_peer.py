import argparse
import logging
import sys
import os
import gym
import gym.wrappers
import numpy
import torch
from torch import nn
import pickle
import agent_factory
from suggestionbuffer import SuggestionBuffer
import shutil

current_dir = os.path.dirname(__file__)
# #TODO: find better solution
# project_base = os.getcwd()
# sys.path.insert(0,project_base)
# import utils as my_utils


import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
from pfrl.agents.ddpg import DDPG
from pfrl.nn import BoundByTanh, ConcatObsAndAction
from pfrl.policies import DeterministicHead

import json
import pybullet_envs
import pybulletgym

def create_dir(folder_name):
    try:
        print(folder_name)
        os.mkdir(folder_name)
    except FileExistsError:
        pass

def save_object(path,obj):
    with open(path, 'wb') as file_pi:
        pickle.dump(obj, file_pi)

def load_object(path):
    with open(path, 'rb') as file_pi2:
        object_pi2 = pickle.load(file_pi2)
        return object_pi2


MIN_EPOCH_LEN =  10000#

IS_LOCAL = False
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
    "ENV" : "HalfCheetahPyBulletEnv-v0"#Pendulum-v0"#""Pendulum-v0"#"Walker2DPyBulletEnv-v0" #"InvertedDoublePendulumPyBulletEnv-v0"#"BipedalWalker-v3"#
}

#value from DDPG paper
gamma = 0.99

parser = argparse.ArgumentParser()

def add_arguments(parser):
    parser.add_argument("--follow-steps",type=int,default=options["FOLLOW_STEPS"])
    parser.add_argument("--hidden-size",type=int,default=options["HIDDEN_SIZE"])
    parser.add_argument("--mix-agents",type=int,default=options["MIX_AGENTS"])
    parser.add_argument("--switch-ratio",type=float,default=options["SWITCH_RATIO"])
    parser.add_argument("--switch-train",type=int,default=options["SWITCH_TRAIN"])
    parser.add_argument("--use-prio",type=int,default=options["USE_PRIO"])
    parser.add_argument("--rbuf-capacity",type=int,default=options["RBUF_CAPACITY"])
    parser.add_argument("--agent-count",type=int,default=options["AGENT_COUNT"])
    parser.add_argument("--use-agent-value",type=int,default=options["USE_AGENT_VALUE"])
    parser.add_argument("--use-trust",type=int,default=options["USE_TRUST"])
    parser.add_argument("--use-trust-buffer",type=int,default=options["USE_TRUST_BUFFER"])
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
    parser.add_argument("--load", type=int, default=options["LOAD"])
    
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

experiment_folder = os.path.join(current_dir,"Experiments",args.save_name)#f"{current_dir}\\Experiments\\{args.save_name}"
ARGS_PATH = os.path.join(experiment_folder,f"{args.save_name}-args.obj")#f"{experiment_folder}\\{args.save_name}_args.obj"
result_folder = os.path.join(experiment_folder,"results.json")#f"{experiment_folder}\\results.json"
save_info_path = os.path.join(experiment_folder,"save_info.json")#f"{experiment_folder}\\results.json"

save_info = [False for x in range(args.agent_count)]

assert(args.peer_learning if args.use_trust else True)
assert(args.peer_learning if args.use_critic else True)
assert(args.peer_learning if args.use_agent_value else True)


create_dir(experiment_folder)

if args.load:
    old_args = load_object(ARGS_PATH)
    old_args.steps = args.steps
    args = old_args


save_object(ARGS_PATH,args)

config_path = os.path.join(experiment_folder,"config.json")#f"{experiment_folder}\\config.json"
with open(config_path, "w") as file:
    json.dump(args.__dict__,file)

# Set a random seed used in PFRL
utils.set_random_seed(args.seed)

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

env = make_env(test=False)
timestep_limit = env.spec.max_episode_steps
obs_space = env.observation_space
action_space = env.action_space

max_episode_len = env.spec.max_episode_steps
max_epoch_len = max(MIN_EPOCH_LEN,max_episode_len)
n_epochs = args.steps // max_epoch_len


if IS_LOCAL:
    print("episodenlänge:",max_episode_len)
    print("epochlänge:",max_epoch_len)

    print("Observation space:", obs_space)
    print("Action space:", action_space)

# CREATE AGENTS
agents = []
sbufs = []

agent_funcs = [
    lambda a,b,c: agent_factory.get_sac_agent(a,b,c),
    lambda a,b,c: agent_factory.get_sac_agent(a,b,c),
    lambda a,b,c: agent_factory.get_sac_agent(a,b,c),
    lambda a,b,c: agent_factory.get_sac_agent(a,b,c),
    lambda a,b,c: agent_factory.get_td3_agent(a,b,c),
    lambda a,b,c: agent_factory.get_ddpg_agent(a,b,c)
    ]
if args.mix_agents:
    agent_funcs = [
    lambda a,b,c: agent_factory.get_sac_agent(a,b,c),
    lambda a,b,c: agent_factory.get_td3_agent(a,b,c),
    lambda a,b,c: agent_factory.get_sac_agent(a,b,c),
    lambda a,b,c: agent_factory.get_td3_agent(a,b,c)
    ]
for i in range(args.agent_count):
    #use the last epoch for trust learning
    sbuf = SuggestionBuffer(max_epoch_len)
    sbufs.append(sbuf)

    #agent = agent_factory.get_td3_agent(obs_space,action_space,args)
    agent = agent_funcs[i](obs_space,action_space,args)

    if args.load:
        #agent.load(experiment_folder+'/agent'+str(i))
        with open(save_info_path, "r") as file:
            save_info = json.load(file)

        if save_info[i]:
            agent.load(os.path.join(experiment_folder,'agent'+str(i)))
            agent.replay_buffer.load(os.path.join(experiment_folder,'agent'+str(i),"buffer"))
        else:
            agent.load(os.path.join(experiment_folder,'agent'+str(i)+"backup"))
            agent.replay_buffer.load(os.path.join(experiment_folder,'agent'+str(i)+'backup',"buffer"))

    agents.append(agent)

agent_values = [200 for _ in range(args.agent_count)]
trust_values = [[200 for _ in range(args.agent_count)] for _ in range(args.agent_count)]
test_rewards = [[] for _ in range(args.agent_count)]
trust_history = [[] for _ in range(args.agent_count)]
follow_history = [[[0 for _ in range(n_epochs)] for _ in range(len(agents))] for _ in range(len(agents))]
stds = [[] for _ in range(args.agent_count)]
start_epoch = 0

DEVICE = "cuda" if args.gpu == 0 else "cpu"
if args.load:
    with open(result_folder, "r") as file:
        results = json.load(file)
        test_rewards = results["rewards"]
        try:
            follow_history = results["follow_history"]
        except:
            pass
        try:
            stds = results["stds"]
        except:
            pass

        try:
            trust_history = results["trust_history"]
        except:
            pass
        start_epoch = results["epoch_count"]
        trust_values = results["trust_values"]   

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


def update_agent_values(agents,agent_values,sbuffers,batch_size=10):
    for buffer in sbuffers:
        if buffer.size < batch_size:
            return

    samples = []

    targets = [0 for _ in range(args.agent_count)]
    counts = [0 for _ in range(args.agent_count)]
    for k in range(args.agent_count):
        samples = sbuffers[k].sample(batch_size//args.agent_count)

        observations = numpy.array([x[3] for x in samples])
        agent = agents[k]
        batch_xs = agent.batch_states(observations, agent.device, agent.phi)

        if isinstance(agent,DDPG):
            vs_ = agent.q_function( (torch.tensor(observations,device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy().astype(float)
        else:
            v1 = agent.q_func1( (torch.tensor(observations,device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy().astype(float)
            v2 = agent.q_func2( (torch.tensor(observations,device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy().astype(float)
            vs_ = numpy.min([v1,v2],axis=0)

        for sample_index in range(batch_size//args.agent_count):
            target = samples[sample_index][0] + gamma*vs_[sample_index][0]
            followed_agent = samples[sample_index][2]
            targets[followed_agent] += target
            counts[followd_agent] += 1
        
    for i in range(args.agent_count):
        target = targets[i] / counts[i]
        v_current = agent_values[i]

        agent_values[i] = (v_current + args.trust_lr *(target - v_current))

def update_trust_values(agent,agent_index,trust_values,sbuffer,batch_size=10):
    samples = sbuffer.sample(batch_size)

    observations = numpy.array([x[3] for x in samples])
    batch_xs = agent.batch_states(observations, agent.device, agent.phi)

    if isinstance(agent,DDPG):
        vs_ = agent.q_function( (torch.tensor(observations,device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy().astype(float)
    else:
        v1 = agent.q_func1( (torch.tensor(observations,device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy().astype(float)
        v2 = agent.q_func2( (torch.tensor(observations,device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy().astype(float)
        vs_ = numpy.min([v1,v2],axis=0)

    targets = [0 for _ in range(args.agent_count)]
    counts = [0 for _ in range(args.agent_count)]
    for sample_index in range(batch_size):
        target = samples[sample_index][0] + gamma*vs_[sample_index]
        followed_agent = samples[sample_index][2]
        counts[followed_agent] += 1
        targets[followed_agent] += target[0]
        
    for i in range(args.agent_count):
        if counts[i] == 0:
            continue
        target = targets[i] / counts[i]
        v_current = trust_values[agent_index][i]

        trust_values[agent_index][i] = (v_current + args.trust_lr *(target - v_current))


def get_action(obs, agent_index, agents, epoch):
    other_indicies = list(filter(lambda x: x != agent_index,numpy.arange(args.agent_count)))
    
    suggested_actions = []
    for i in other_indicies:
        with agents[i].eval_mode():
            _action = agents[i].act(obs)
            suggested_actions.append((i,_action))

    agent = agents[agent_index]

    own_action = agent.act(obs)

    #TODO How to combine agent and trust values?
    if args.use_agent_value or args.use_trust or args.use_critic:
        vals = numpy.zeros((args.agent_count))
        if args.use_trust:
            max_val = numpy.max(numpy.abs(trust_values[agent_index]))
            normalized = numpy.array(trust_values[agent_index]) / max_val
            vals = vals + normalized
            
        if args.use_agent_value:
            max_val = numpy.max(numpy.abs(agent_values))
            normalized = numpy.array(agent_values) / max_val
            vals = vals + normalized

        if args.use_critic:
            critic = numpy.zeros((args.agent_count))
            for i in range(args.agent_count):
                batch_xs = agent.batch_states([obs], agent.device, agent.phi)

                action_to_evaluate = own_action if agent_index == i else list(filter(lambda x: x[0]==i,suggested_actions))[0][1]
                action_to_evaluate = torch.tensor([action_to_evaluate],device=DEVICE)
                if isinstance(agent,DDPG):
                    v_ = agent.q_function( (torch.tensor([obs],device=DEVICE),action_to_evaluate) ).cpu().detach().numpy()[0][0]
                else:
                    v1 = agent.q_func1( (torch.tensor([obs],device=DEVICE),action_to_evaluate) ).cpu().detach().numpy()[0][0]
                    v2 = agent.q_func2( (torch.tensor([obs],device=DEVICE),action_to_evaluate) ).cpu().detach().numpy()[0][0]
                    v_ = min(v1,v2)

                critic[i] = v_
            max_critic = numpy.max(numpy.abs(critic))
            normalized = critic / max_critic
            vals = vals + normalized

        if numpy.min(vals) < 0:
            vals = vals + numpy.abs(numpy.min(vals))
        
        temperature = args.T*numpy.exp(-args.T_decay*epoch)
        probabilities = numpy.exp(vals/temperature) / numpy.sum(numpy.exp(vals/temperature))
        followed_agent = numpy.random.choice(numpy.arange(args.agent_count),p=probabilities)
        action = own_action if agent_index == followed_agent else list(filter(lambda x: x[0]==followed_agent,suggested_actions))[0][1]

        return action,[followed_agent]

    
    critic = numpy.zeros((args.agent_count))
    for i in range(args.agent_count):
        batch_xs = agent.batch_states([obs], agent.device, agent.phi)
        v_ = agent.q_function( (torch.tensor([obs],device=DEVICE),agents[i].policy(batch_xs).sample()) ).cpu().detach().numpy()[0][0]
        critic[i] = v_
    followed_agent = numpy.argmax(critic)
    action = own_action if agent_index == followed_agent else list(filter(lambda x: x[0]==followed_agent,suggested_actions))[0][1]

    return action,[followed_agent]


boost_single = False
if (args.switch_ratio) < 1:
    boost_single = True
    args.switch_ratio = round(1 / args.switch_ratio)

for i in range(start_epoch + 1, n_epochs + 1):
    # 0.5  => 2x so oft single
    # 2    => 2x so oft peer
    # 3    => 3x so oft peer
    # 0.33 => 3x so oft single   
    single_epoch = i%(1+args.switch_ratio)==1 if args.switch_train else False
    if boost_single and args.switch_train:
        single_epoch = not single_epoch


    for k in range(len(agents)):
        agent = agents[k]
        obs = env.reset()
        obs2 = env.reset()
        R = 0  # return (sum of rewards)
        t = 0  # time step
        epoch_steps = 0
        current_follow_step = 0
        while True:
           #Uncomment to watch the behavior in a GUI window
            if args.render_train:
                env.render()

            if args.peer_learning and not single_epoch:
                if current_follow_step == 0:
                    action, followed_agents = get_action(obs,k,agents,i)
                else:
                    with agents[followed_agents[0]].eval_mode():
                        action = agents[followed_agents[0]].act(obs)
                    current_follow_step += 1
                    if current_follow_step == args.follow_steps:
                        current_follow_step = 0
            else:
                action = agent.act(obs)
            
            obs, reward, done, info = env.step(action)
            
            R += reward
            t += 1
            epoch_steps += 1
            reset = t == max_episode_len

            if args.peer_learning and not single_epoch:
                follow_history[k][followed_agents[0]][i-1] = follow_history[k][followed_agents[0]][i-1] + 1

                for followed_agent in followed_agents:
                    sbufs[k].add(reward,action,followed_agent,[x for x in obs])

                    if args.use_agent_value:
                        update_agent_values(agents,agent_values,sbufs)
                        #agent_values[followed_agent] = agent_values[followed_agent] + args.trust_lr * (reward + gamma*v_ - agent_values[followed_agent])

                                        
                    if args.use_trust and not args.use_trust_buffer:
                        batch_xs = agent.batch_states([obs], agent.device, agent.phi)
                        if isinstance(agent,DDPG):
                            v_ = agent.q_function( (torch.tensor([obs],device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy()[0][0]
                        else:
                            v1 = agent.q_func1( (torch.tensor([obs],device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy()[0][0]
                            v2 = agent.q_func2( (torch.tensor([obs],device=DEVICE),agent.policy(batch_xs).sample()) ).cpu().detach().numpy()[0][0]

                            v_ = min(v1,v2)

                        trust_values[k][followed_agent] = trust_values[k][followed_agent] + args.trust_lr * (reward + gamma*v_ - trust_values[k][followed_agent])

            #check if this is sufficient
            agent.batch_last_action = [action]
            agent.observe(obs, reward, done, reset)

            if done or reset or epoch_steps >= max_epoch_len:
                obs = env.reset()
                R = 0  # return (sum of rewards)
                t = 0  # time step
                if epoch_steps >= max_epoch_len:
                    break
        
        if args.use_trust_buffer and not single_epoch:
            for _ in range(max_epoch_len):
                update_trust_values(agent,k,trust_values,sbufs[k],10)            
            trust_history[k].append([x for x in trust_values[k]])

        if i % args.eval_interval == 0:
            if IS_LOCAL:
                print("agent",k)
                print('epoch:', i, 'R:', "agent_values", agent_values)
                print('epoch:', i, 'R:', "trust_values", trust_values[k])
            agent.save(experiment_folder+'/agent'+str(k))

            agent.save(os.path.join(experiment_folder,'agent'+str(k)+'backup'))
            agent.replay_buffer.save(os.path.join(experiment_folder,'agent'+str(k)+'backup',"buffer"))

            save_info[k] = False
            with open(save_info_path, "w") as file:
                    json.dump(save_info,file)

            agent.save(os.path.join(experiment_folder,'agent'+str(k)))
            agent.replay_buffer.save(os.path.join(experiment_folder,'agent'+str(k),"buffer"))

            save_info[k] = True
            with open(save_info_path, "w") as file:
                    json.dump(save_info,file)
            
            
            reward, std = test_agent(agent,env,max_episode_len)
            test_rewards[k].append(reward)
            stds[k].append(std)

            if k == args.agent_count - 1 :
                with open(result_folder, "w") as file:
                    json.dump({
                        "epoch_count" : i,
                        "trust_values" : trust_values,
                        "follow_history" : follow_history,
                        "rewards" : test_rewards,
                        "stds" : stds,
                        "trust_history" : trust_history
                    },file)

for k in range(args.agent_count):
    if save_info[k]:
        shutil.rmtree(os.path.join(experiment_folder,'agent'+str(k)+'backup'), ignore_errors=False, onerror=None)
    else:
        shutil.rmtree(os.path.join(experiment_folder,'agent'+str(k)), ignore_errors=False, onerror=None)

        

print('Finished.')
