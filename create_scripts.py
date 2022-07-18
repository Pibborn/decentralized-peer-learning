import re
from pathlib import Path
def run():
    # Writing to file
    script_folder = 'scripts_change_parameter'
    output_folder = 'output'
    environments = ['HalfCheetahBulletEnv-v0', 'HalfCheetahPyBulletEnv-v0',
                   'ReacherBulletEnv-v0', 'HopperPyBulletEnv-v0']

    number_agent = [4, 1]
    runnning_time = {
        'HalfCheetahBulletEnv-v0': '96:00:00',
        'HalfCheetahPyBulletEnv-v0': '96:00:00',
        'ReacherBulletEnv-v0': '24:00:00',
        'HopperPyBulletEnv-v0': '96:00:00'
    }
    scripts = {
        'peer': 'run_peer.py',
        'full_info': 'run_fullinfo.py'
    }
    learning_rate = {'exponential_decay': "\"lambda x: (np.exp(x) - 1) * 7.3e-4 + 7.3e-4\" " \
                                          "\"lambda x: (np.exp(x) - 1) * 7.3e-4 + 7.3e-4\" " \
                                          "\"lambda x: (np.exp(x) - 1) * 7.3e-4 + 7.3e-4\" " \
                                          "\"lambda x: (np.exp(x) - 1) * 7.3e-4 + 7.3e-4\" "
                     }
    switch_ratio = 1
    mix_agents = ["\"SAC SAC SAC SAC\""]# "\"SAC SAC TD3 TD3\"", "\"TD3 TD3 TD3 TD3\""]
    net_archs = ["\"25 25\"", "\"150 200\"", "\"200 300\"", "\"350 300\""]
    n_timesteps = {
        'HalfCheetahBulletEnv-v0': 1_000_000,
        'HalfCheetahPyBulletEnv-v0': 1_000_000,
        'ReacherBulletEnv-v0': 300_000,
        'HopperPyBulletEnv-v0': 1_000_000
    }
    buffer_size = {
        'HalfCheetahBulletEnv-v0': 300_000,
        'HalfCheetahPyBulletEnv-v0': 300_000,
        'ReacherBulletEnv-v0': 300_000,
        'HopperPyBulletEnv-v0': 300_000
    }
    experiment_list = []
    for agent_type, script_name in scripts.items():
        for env in environments:
            for a_num in number_agent:
                for mix in mix_agents:
                    for net_arch in net_archs:
                        for learning_rate_key in learning_rate.keys():
                            experiment_name = create_script(agent_type, env, learning_rate, learning_rate_key,
                                                                 mix, a_num, runnning_time, scripts,
                                                                 switch_ratio, net_arch, script_folder,
                                                            n_timesteps, buffer_size)
                            experiment_list.append(experiment_name)

    write_experiment_names_to_file(experiment_list, script_folder)
    path = Path(f"{script_folder}/{output_folder}")
    path.mkdir(exist_ok=True)
    write_sbatch_comments_to_file(experiment_list, script_folder)

def create_script(agent_type, enviroments, learning_rate, learning_rate_key, mix_agents, number_agent,
                       runnning_time, scripts, switch_ratio, net_arch, script_folder,
                  n_timesteps, buffer_size):
    experiment_name = f'{agent_type}_{number_agent}_{enviroments}_{net_arch}_{mix_agents}_{learning_rate_key}'
    experiment_name = re.sub("\"", "",experiment_name)
    experiment_name = re.sub(" ", "_", experiment_name)

    net_arch = net_arch.replace("\"", "")
    mix_agents = mix_agents.replace("\"", "")
    with open(f"{script_folder}/{experiment_name}.sh", "w") as file1:
        # Writing data to a file
        file1.write(f"#!/bin/bash\n")
        write_SBATCH_commants(enviroments, experiment_name, file1, runnning_time)
        write_prepare_enviroment(file1)

        write_python_default_parameter(agent_type, enviroments, file1, learning_rate, learning_rate_key,
                                       net_arch, number_agent, scripts, n_timesteps, buffer_size)
        if agent_type == 'peer':
            add_peer_arguments( file1, switch_ratio, mix_agents)
        elif agent_type == 'full_info':
            add_full_info_arguments(file1, mix_agents)

    return experiment_name


def write_prepare_enviroment(file1):
    file1.writelines("module purge \n")
    file1.writelines("module load devel/SWIG/4.0.1-foss-2019b-Python-3.7.4\n")
    file1.writelines("module load devel/PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4\n")
    file1.writelines("cd ..\n")
    file1.writelines("export PYTHONPATH=$PYTHONPATH:$(pwd)\n")
    file1.writelines("export http_proxy=http://webproxy.zdv.uni-mainz.de:8888\n")
    file1.writelines("export https_proxy=https://webproxy.zdv.uni-mainz.de:8888\n")
    file1.writelines("source virt/bin/activate\n")
    file1.writelines("wandb offline\n")


def write_SBATCH_commants(enviroments, experiment_name, file1, runnning_time):
    file1.write(f"#SBATCH --job-name={experiment_name}\n")
    file1.writelines("#SBATCH -p smp \n")
    file1.writelines("#SBATCH --account=m2_datamining \n")
    file1.writelines(f"#SBATCH --time={runnning_time[enviroments]} \n")
    file1.writelines("#SBATCH --tasks=1 \n")
    file1.writelines("#SBATCH --nodes=1 \n")
    file1.writelines("#SBATCH --cpus-per-task=10 \n")
    file1.writelines("#SBATCH --mem=10G \n")
    file1.writelines("\n")
    file1.writelines("#SBATCH -o \%x_\%j_profile.out \n")
    file1.writelines("#SBATCH -C anyarch \n")
    file1.writelines("#SBATCH -o output_8_06_22/\%x_\%j.out \n")
    file1.writelines("#SBATCH -e output_8_06_22/\%x_\%j.err\n")
    file1.writelines("#SBATCH --mail-user=bruggerj@uni-mainz.de\n")
    file1.writelines("#SBATCH --mail-type=FAIL \n")
    file1.writelines("#SBATCH --mail-type=END\n")
    file1.writelines("\n")


def write_python_default_parameter(agent_type, enviroments, file1, learning_rate, learning_rate_key,
                                   net_arch, number_agent, scripts, n_timesteps, buffer_size):
    file1.writelines(f"srun python {scripts[agent_type]} \\\n")
    file1.writelines(f"  --save-name $SLURM_JOB_NAME \\\n")
    file1.writelines(f"  --job_id $SLURM_JOB_ID \\\n")
    file1.writelines(f"  --env {enviroments} \\\n")
    file1.writelines(f"  --agent-count {number_agent} \\\n")
    file1.writelines(f"  --batch-size 256 \\\n")
    file1.writelines(f"  --buffer-size {buffer_size[enviroments]} \\\n")
    file1.writelines(f"  --steps {n_timesteps[enviroments]} \\\n")
    file1.writelines(f"  --buffer-start-size 10_000 \\\n")
    file1.writelines(f"  --gamma 0.98 \\\n")
    file1.writelines(f"  --gradient_steps 8 \\\n")
    file1.writelines(f"  --tau 0.02 \\\n")
    file1.writelines(f"  --train_freq 8 \\\n")
    file1.writelines(f"  --seed $SLURM_JOB_ID \\\n")
    file1.writelines(f"  --net-arch {net_arch} \\\n")
    file1.writelines(f"  --learning_rate {learning_rate[learning_rate_key]} \\\n")



def add_peer_arguments( file1, switch_ratio, mix_agents):
    file1.writelines(f"  --mix-agents {mix_agents}  \\\n")
    file1.writelines(f"  --switch-ratio {switch_ratio} \\\n")
    file1.writelines(f"--use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise False \n")


def add_full_info_arguments(file1, mix_agents):
    file1.writelines(f"  --mix-agents {mix_agents}  \n")
    file1.writelines(f"\n")
    file1.writelines(f"\n")


def write_experiment_names_to_file(experiment_list,script_folder ):
    with open(f"{script_folder}/experiment_name.txt", "w") as file2:
        for experiment_name in experiment_list:
            file2.write(f"{experiment_name}\n")

def write_sbatch_comments_to_file(experiment_list,script_folder ):
    with open(f"{script_folder}/sbatch_comments.txt", "w") as file2:
        for experiment_name in experiment_list:
            file2.write(f"sbatch {experiment_name}.sh \n")



if __name__ == '__main__':
    run()
