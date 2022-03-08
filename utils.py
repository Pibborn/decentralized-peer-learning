import datetime

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
    "ENV" : "HalfCheetahPyBulletEnv-v0",#Pendulum-v0"#""Pendulum-v0"#"Walker2DPyBulletEnv-v0" #"InvertedDoublePendulumPyBulletEnv-v0"#"BipedalWalker-v3"#
    "N_AGENTS": 4,
    "BASESAVELOC": "./agents/",
    "SAVEDIR": datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
    "LOGINTERVAL": 1000
}

def check_args(args):
    assert(args.peer_learning if args.use_trust else True)
    assert(args.peer_learning if args.use_critic else True)
    assert(args.peer_learning if args.use_agent_value else True)


def constant_schedule(lr):

    def func(progress_remaining):
        return lr

    return func
