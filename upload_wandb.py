import os
import glob
from pathlib import Path
import json


def run():
    directory = [
        "peer_4_HalfCheetah-v4_200_300_adv_True_T_1",
        "peer_4_HalfCheetah-v4_200_300_adv_False_T_1",
        "peer_1_HalfCheetah-v4_200_300_adv_True_T_1",
        "peer_1_HalfCheetah-v4_200_300_adv_False_T_1",
        "peer_4_Walker2d-v4_200_300_adv_True_T_1",
        "peer_4_Walker2d-v4_200_300_adv_False_T_1",
        "peer_1_Walker2d-v4_200_300_adv_True_T_1",
        "peer_1_Walker2d-v4_200_300_adv_False_T_1",
        "peer_4_Ant-v4_200_300_adv_True_T_1",
        "peer_4_Ant-v4_200_300_adv_False_T_1",
        "peer_1_Ant-v4_200_300_adv_True_T_1",
        "peer_1_Ant-v4_200_300_adv_False_T_1",
        "peer_4_Hopper-v4_200_300_adv_True_T_1",
        "peer_4_Hopper-v4_200_300_adv_False_T_1",
        "peer_1_Hopper-v4_200_300_adv_True_T_1",
        "peer_1_Hopper-v4_200_300_adv_False_T_1",
        "peer_4_Swimmer-v4_200_300_adv_True_T_1",
        "peer_4_Swimmer-v4_200_300_adv_False_T_1",
        "peer_1_Swimmer-v4_200_300_adv_True_T_1",
        "peer_1_Swimmer-v4_200_300_adv_False_T_1",
        "peer_4_InvertedDoublePendulum-v4_200_300_adv_True_T_1",
        "peer_4_InvertedDoublePendulum-v4_200_300_adv_False_T_1",
        "peer_1_InvertedDoublePendulum-v4_200_300_adv_True_T_1",
        "peer_1_InvertedDoublePendulum-v4_200_300_adv_False_T_1",
        "full_info_4_HalfCheetah-v4_200_300_T_1",
        "full_info_1_HalfCheetah-v4_200_300_T_1",
        "full_info_4_Walker2d-v4_200_300_T_1",
        "full_info_1_Walker2d-v4_200_300_T_1",
        "full_info_4_Ant-v4_200_300_T_1",
        "full_info_1_Ant-v4_200_300_T_1",
        "full_info_4_Hopper-v4_200_300_T_1",
        "full_info_1_Hopper-v4_200_300_T_1",
        "full_info_4_Swimmer-v4_200_300_T_1",
        "full_info_1_Swimmer-v4_200_300_T_1",
        "full_info_4_InvertedDoublePendulum-v4_200_300_T_1",
        "full_info_1_InvertedDoublePendulum-v4_200_300_T_1",

    ]
    wandb_entity_name = 'jgu-wandb'
    wandb_project = 'peer-learning'
    upload_identifier = ""
    Path_to_experiments = Path("/home/jbrugger/PycharmProjects/decentralized-peer-learning/Experiments")

    for setup in directory:
        print(f"-------------------{setup}")
        upload_experiment(Path_to_experiments, setup, upload_identifier, wandb_entity_name, wandb_project)


def upload_experiment(Path_to_experiments, setup, upload_identifier, wandb_entity_name, wandb_project):
    path_to_runs = Path_to_experiments / setup
    list_of_runs = glob.glob(f"{path_to_runs}/*")  # * means all if need specific format then *.csv
    runs = get_all_results(list_of_runs)
    # runs = get_run_with_most_results(list_of_runs)
    for run in runs:
        try:
            path_to_run_to_upload = path_to_runs / run / 'wandb'
            id = f"{setup}__{run}{upload_identifier}"
            path_to_run_to_upload = get_path_to_wandb_file(path_to_run_to_upload)
            if len(id) >= 120:
                 id = id.replace('learning_rate', 'lr')
                 id = id.replace('temperature_decay', 'td')
            command = f"wandb sync -e {wandb_entity_name} " \
                      f"-p {wandb_project}" \
                      f" --id {id}" \
                      f" {path_to_run_to_upload}"
            print(command)
            os.system(command)
        except FileNotFoundError:
            print(f"skiped {run}")


def get_path_to_wandb_file(path_to_run_to_upload):
    files_in_wandb_folder = glob.glob(f"{path_to_run_to_upload}/*")
    for file_in_wandb in files_in_wandb_folder:
        name_file = Path(file_in_wandb).name
        if 'offline-run' in name_file:
            path_to_run_to_upload = path_to_run_to_upload / name_file
    return path_to_run_to_upload


def check_if_multi_processing_was_used(path_to_log_file):
    with open(path_to_log_file) as f:
        data = list(json.load(f).items())
        for t in data:
            if t[0] == "args":
                for entry in t[1]:
                    if entry == '--multi-threading':
                        return True
    return False


def check_if_CAT_was_used(path_to_log_file):
    with open(path_to_log_file) as f:
        data = list(json.load(f).items())
        for t in data:
            if t[0] == "args":
                for entry in t[1]:
                    if entry == '--multi-threading':
                        return True
    return False


def get_all_results(list_of_runs):
    return [Path(run_path).name for run_path in list_of_runs]


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
