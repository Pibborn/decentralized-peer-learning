import os
import glob
from pathlib import Path


def run():
    directory = [
        'dictator_new_1_agent_Walker2DPyBulletEnv-v0',
        'dictator_new_2_agent_HalfCheetahBulletEnv-v0',
        'dictator_new_2_agent_Pendulum-v0',
        'dictator_new_2_agent_Walker2DPyBulletEnv-v0',
        'dictator_new_3_agent_HalfCheetahBulletEnv-v0',
        'dictator_new_3_agent_Pendulum-v0',
        'dictator_new_3_agent_Walker2DPyBulletEnv-v0',
        'dictator_new_4_agent_HalfCheetahBulletEnv-v0',
        'dictator_new_4_agent_Pendulum-v0',
        'dictator_new_4_agent_Walker2DPyBulletEnv-v0',
        'peer_1_agent_HalfCheetahBulletEnv-v0',
        'peer_1_agent_Pendulum-v0',
        'peer_1_agent_Walker2DPyBulletEnv-v0',
        'peer_2_agent_HalfCheetahBulletEnv-v0',
        'peer_2_agent_Pendulum-v0',
        'peer_2_agent_Walker2DPyBulletEnv-v0',
        'peer_3_agent_HalfCheetahBulletEnv-v0',
        'peer_3_agent_Pendulum-v0',
        'peer_3_agent_Walker2DPyBulletEnv-v0',
        'peer_4_agent_HalfCheetahBulletEnv-v0',
        'peer_4_agent_Pendulum-v0',
        'peer_4_agent_Walker2DPyBulletEnv-v0',
    ]
    wandb_entity_name = 'jgu-wandb'
    wandb_project = 'peer-learning'
    upload_identifier = "4.04.2022"
    Path_to_experiments = Path("/home/jbrugger/PycharmProjects/decentralized-peer-learning/Experiments")

    print('\n')
    for setup in directory:
        upload_experiment(Path_to_experiments, setup, upload_identifier, wandb_entity_name, wandb_project)


def upload_experiment(Path_to_experiments, setup, upload_identifier, wandb_entity_name, wandb_project):
    path_to_runs = Path_to_experiments / setup
    list_of_runs = glob.glob(f"{path_to_runs}/*")  # * means all if need specific format then *.csv
    runs = get_run_with_most_results(list_of_runs)
    for run in runs:
        path_to_run_to_upload = path_to_runs / run
        id = f"{setup}__{run}__{upload_identifier}"
        command = f"wandb sync -e {wandb_entity_name} " \
              f"-p {wandb_project}" \
              f" --id {id}" \
              f" {path_to_run_to_upload}"
        print(command)
        os.system(command)


def get_run_with_most_results(list_of_runs):
    run_with_most_results = ''
    max_number_of_results = 0
    for run_path in list_of_runs:
        run_name = Path(run_path).name
        path_to_peer = Path(run_path) / 'Peer0_0'
        number_of_results = len(glob.glob(f"{path_to_peer}/*"))
        if number_of_results > max_number_of_results:
            max_number_of_results = number_of_results
            run_with_most_results = run_name
    return [run_with_most_results]


if __name__ == '__main__':
    run()
