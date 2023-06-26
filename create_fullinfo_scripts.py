import re
from pathlib import Path


def run():
    # Writing to file
    script_folder = 'scripts_smaller_net_mujoco'
    output_folder = 'output'
    environments = ['HalfCheetah-v4',
                    'Walker2d-v4',
                    'Ant-v4',
                    'Hopper-v4',


                    # 'HalfCheetahBulletEnv-v0',
                    # 'HalfCheetahPyBulletEnv-v0'
                    # 'ReacherBulletEnv-v0',
                    # 'HopperPyBulletEnv-v0'
                    ]
    env_args = {'HalfCheetah-v4': '',
                'Walker2d-v4': 'terminate_when_unhealthy=False',
                'Ant-v4': 'terminate_when_unhealthy=False',
                'Hopper-v4': 'terminate_when_unhealthy=False',
                'Swimmer-v4': '',
                'InvertedDoublePendulum-v4': '',

                'HalfCheetahBulletEnv-v0': "",
                'HalfCheetahPyBulletEnv-v0': "",
                'ReacherBulletEnv-v0': "",
                'HopperPyBulletEnv-v0': ""
                }

    number_agent = [1]
    runnning_time = {
        'HalfCheetah-v4': '72:00:00',
        'Walker2d-v4': '72:00:00',
        'Ant-v4': '72:00:00',
        'Hopper-v4': '72:00:00',
        'Swimmer-v4': '72:00:00',
        'InvertedDoublePendulum-v4': '24:00:00',
        'HalfCheetahBulletEnv-v0': '72:00:00',
        'HalfCheetahPyBulletEnv-v0': '72:00:00',
        'ReacherBulletEnv-v0': '24:00:00',
        'HopperPyBulletEnv-v0': '24:00:00'
    }
    scripts = {
        'full_info': 'run_fullinfo.py'
    }
    learning_rate = {
        'HalfCheetah-v4': "\"lambda x: 7.3e-4\"",
        'Walker2d-v4': "\"lambda x: 3e-4\"",
        'Ant-v4': "\"lambda x: 3e-4\"",
        'Hopper-v4': "\"lambda x: 3e-4\"",
        'Swimmer-v4': "\"lambda x: 3e-4\"",
        'InvertedDoublePendulum-v4': "\"lambda x: 3e-4\"",

        'HalfCheetahBulletEnv-v0': "\"lambda x: 7.3e-4\"",
        'HalfCheetahPyBulletEnv-v0': "\"lambda x: 7.3e-4\"",
        'ReacherBulletEnv-v0': "\"lambda x: 3e-4\"",
        'HopperPyBulletEnv-v0': "\"lambda x: 3e-4\""
    }
    # learning_rate = {'7.3e-4': 7.3e-4}
    mix_agents = ["\"SAC\""]  # ["\"SAC SAC SAC SAC\""]# "\"SAC SAC TD3 TD3\"", "\"TD3 TD3 TD3 TD3\""]
    net_archs = [ "\"25 25\"", "\"150 200\"", "\"200 300\"", "\"350 300\""]
    n_timesteps = {
        'HalfCheetah-v4': 1_000_000,
        'Walker2d-v4': 1_000_000,
        'Ant-v4': 1_000_000,
        'Hopper-v4': 1_000_000,
        'Swimmer-v4': 400_000,
        'InvertedDoublePendulum-v4': 500_000,

        'HalfCheetahBulletEnv-v0': 1_000_000,
        'HalfCheetahPyBulletEnv-v0': 1_000_000,
        'ReacherBulletEnv-v0': 150_000,
        'HopperPyBulletEnv-v0': 1_000_000
    }
    buffer_size = {
        'HalfCheetah-v4': 1_000_000,
        'Walker2d-v4': 1_000_000,
        'Ant-v4': 1_000_000,
        'Hopper-v4': 1_000_000,
        'Swimmer-v4': 1_000_000,
        'InvertedDoublePendulum-v4': 500_000,

        'HalfCheetahBulletEnv-v0': 300_000,
        'HalfCheetahPyBulletEnv-v0': 300_000,
        'ReacherBulletEnv-v0': 300_000,
        'HopperPyBulletEnv-v0': 300_000
    }

    buffer_start_size = {
        'HalfCheetah-v4': 10_000,
        'Walker2d-v4': 100,
        'Ant-v4': 100,
        'Hopper-v4': 100,
        'Swimmer-v4': 100,
        'InvertedDoublePendulum-v4': 100,

        'HalfCheetahBulletEnv-v0': 10_000,
        'HalfCheetahPyBulletEnv-v0': 10_000,
        'ReacherBulletEnv-v0': 100,
        'HopperPyBulletEnv-v0': 100
    }
    gamma = {
        'HalfCheetah-v4': 0.99,
        'Walker2d-v4': 0.99,
        'Ant-v4': 0.99,
        'Hopper-v4': 0.99,
        'Swimmer-v4': 0.999,
        'InvertedDoublePendulum-v4': 0.99,

        'HalfCheetahBulletEnv-v0': 0.99,
        'HalfCheetahPyBulletEnv-v0': 0.99,
        'ReacherBulletEnv-v0': 0.99,
        'HopperPyBulletEnv-v0': 0.99
    }

    epsilion_list = [0.2]
    temperature_list = [1]
    sample_from_suggestions_list = [True]
    experiment_list = []
    follow_steps_list = [10]

    path = Path(f"{script_folder}/{output_folder}")
    path.mkdir(exist_ok=True, parents=True)

    for agent_type, script_name in scripts.items():
        for env in environments:
            for a_num in number_agent:
                for mix in mix_agents:
                    for net_arch in net_archs:
                        for epsilon in epsilion_list:
                            for sample_from_suggestions in sample_from_suggestions_list:

                                    for temperature in temperature_list:
                                        experiment_name = create_script(agent_type, env, learning_rate, mix, a_num,
                                                                        runnning_time, scripts, net_arch, script_folder,
                                                                        n_timesteps, buffer_size, temperature,
                                                                        output_folder, env_args,
                                                                        buffer_start_size, gamma)
                                        experiment_list.append(experiment_name)


    write_experiment_names_to_file(experiment_list, script_folder)
    write_sbatch_comments_to_file(experiment_list, script_folder)


def create_script(agent_type, enviroments, learning_rate, mix_agents, number_agent, runnning_time, scripts, net_arch,
                  script_folder, n_timesteps, buffer_size, temperature, output_folder,  env_args,
                  buffer_start_size, gamma):
    experiment_name = f'{agent_type}_{number_agent}_{enviroments}_{net_arch}' \
                      f'_T_{temperature}'

    # f'_{mix_agents}_{learning_rate_key}' \f'_sr_{switch_ratio}_fs_{follow_steps}' \

    experiment_name = re.sub("\"", "", experiment_name)
    experiment_name = re.sub(" ", "_", experiment_name)

    net_arch = net_arch.replace("\"", "")
    mix_agents = mix_agents.replace("\"", "")
    with open(f"{script_folder}/{experiment_name}.sh", "w") as file1:
        # Writing data to a file
        file1.write(f"#!/bin/bash\n")
        write_SBATCH_commants(enviroments, experiment_name, file1, runnning_time, output_folder)
        write_prepare_enviroment(file1)

        write_python_default_parameter(agent_type=agent_type, enviroments=enviroments, file1=file1,
                                       learning_rate=learning_rate, net_arch=net_arch, number_agent=number_agent,
                                       scripts=scripts, n_timesteps=n_timesteps, buffer_size=buffer_size,
                                       env_args=env_args, buffer_start_size=buffer_start_size, gamma=gamma)

        add_full_info_arguments(file1, mix_agents)

    return experiment_name


def write_prepare_enviroment(file1):
    file1.writelines("module purge \n")
    file1.writelines("module load devel/PyTorch/1.9.0-fosscuda-2020b\n")
    file1.writelines("cd ..\n")
    file1.writelines("export PYTHONPATH=$PYTHONPATH:$(pwd)\n")
    file1.writelines("export http_proxy=http://webproxy.zdv.uni-mainz.de:8888\n")
    file1.writelines("export https_proxy=https://webproxy.zdv.uni-mainz.de:8888\n")
    file1.writelines("source venv386/bin/activate\n")
    file1.writelines("wandb offline\n")


def write_SBATCH_commants(enviroments, experiment_name, file1, runnning_time, output_folder):
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
    file1.writelines(f"#SBATCH -o {output_folder}/\%x_\%j.out \n")
    file1.writelines(f"#SBATCH -e {output_folder}/\%x_\%j.err\n")
    file1.writelines("#SBATCH --mail-user=bruggerj@uni-mainz.de\n")
    file1.writelines("#SBATCH --mail-type=FAIL \n")
    file1.writelines("#SBATCH --mail-type=END\n")
    file1.writelines("\n")


def write_python_default_parameter(agent_type, enviroments, file1, learning_rate, net_arch, number_agent, scripts,
                                   n_timesteps, buffer_size, env_args, buffer_start_size, gamma):
    file1.writelines(f"srun python {scripts[agent_type]} \\\n")
    file1.writelines(f"  --save-name $SLURM_JOB_NAME \\\n")
    file1.writelines(f"  --job_id $SLURM_JOB_ID \\\n")
    file1.writelines(f"  --env {enviroments} \\\n")
    file1.writelines(f"  --env_args {env_args[enviroments]} \\\n")
    file1.writelines(f"  --agent-count {number_agent} \\\n")
    file1.writelines(f"  --batch-size 256 \\\n")
    file1.writelines(f"  --buffer-size {buffer_size[enviroments]} \\\n")
    file1.writelines(f"  --steps {n_timesteps[enviroments]} \\\n")
    file1.writelines(f"  --buffer-start-size {buffer_start_size[enviroments]} \\\n")
    file1.writelines(f"  --gamma {gamma[enviroments]} \\\n")
    file1.writelines(f"  --gradient_steps 8 \\\n")
    file1.writelines(f"  --tau 0.02 \\\n")
    file1.writelines(f"  --train_freq 8 \\\n")
    file1.writelines(f"  --seed $SLURM_JOB_ID \\\n")
    file1.writelines(f"  --net-arch {net_arch} \\\n")
    file1.writelines(f"  --learning_rate {learning_rate[enviroments]} \\\n")



def add_full_info_arguments(file1, mix_agents):
    file1.writelines(f"  --mix-agents {mix_agents}  \n")
    file1.writelines(f"\n")
    file1.writelines(f"\n")


def write_experiment_names_to_file(experiment_list, script_folder):
    with open(f"{script_folder}/experiment_name_fullinfo.txt", "w") as file2:
        for experiment_name in experiment_list:
            file2.write(f"\"{experiment_name}\",\n")


def write_sbatch_comments_to_file(experiment_list, script_folder):
    with open(f"{script_folder}/sbatch_comments_fullinfo.txt", "w") as file2:
        for experiment_name in experiment_list:
            file2.write(f"sbatch {experiment_name}.sh \n")


if __name__ == '__main__':
    run()
