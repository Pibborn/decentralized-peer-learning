import os
import glob
from pathlib import Path
import json
import datetime
from subprocess import STDOUT, check_output, TimeoutExpired

def run():
    directory = [
        #"peer_4_HalfCheetah-v4_adv_False__peer_noise_sample_True__early_learning",
        #"peer_4_Hopper-v4_adv_False__peer_noise_sample_True__early_learning",
        "peer_4_Walker2d-v4_adv_False__peer_noise_sample_True__early_learning"

        #"peer_3_Room-v21_adv_True_ACT_adversarial"
        #"peer_4_Ant-v4_150_200__adv_True_ACT_",
        # "peer_4_Ant-v4_150_200__adv_True_AT_",
        # "peer_4_Ant-v4_150_200__adv_True_CT_",
        #"peer_4_Ant-v4_150_200__adv_True_T_",
        # "peer_4_Ant-v4_150_200__adv_True_AC_",
        # "peer_4_Ant-v4_150_200__adv_True_A_",
        # "peer_4_Ant-v4_150_200__adv_True_C_",
        #
        # "peer_4_Hopper-v4_150_200__adv_True_ACT_",
        # "peer_4_Hopper-v4_150_200__adv_True_AT_",
        # "peer_4_Hopper-v4_150_200__adv_True_CT_",
        # "peer_4_Hopper-v4_150_200__adv_True_T_",
        # "peer_4_Hopper-v4_150_200__adv_True_AC_",
        # "peer_4_Hopper-v4_150_200__adv_True_A_",
        #"peer_4_Hopper-v4_150_200__adv_True_C_",
        ## "peer_2_Room-v21_adv_True_ACT_adversarial",
        ## "peer_2_Room-v21_adv_False_ACT_adversarial",
        ## "peer_2_Room-v21_adv_False__random_follow",
        #
        ## "peer_3_Room-v21_adv_False__random_follow",
        #
        #"peer_2_Room-v21_adv_False__peer_noise_sample_True__mel_0early_learning",

        ## "peer_3_Room-v21_adv_True_ACT_adversarial",
        # "peer_3_Room-v21_adv_True_AT_adversarial",
        # "peer_3_Room-v21_adv_True_CT_adversarial",
        # "peer_3_Room-v21_adv_True_T_adversarial",
        # "peer_3_Room-v21_adv_True_AC_adversarial",
        # "peer_3_Room-v21_adv_True_A_adversarial",
        # "peer_3_Room-v21_adv_True_C_adversarial",
        # "peer_3_Room-v21_adv_False_ACT_adversarial",
        # "peer_3_Room-v21_adv_False_AT_adversarial",
        # "peer_3_Room-v21_adv_False_CT_adversarial",
        # "peer_3_Room-v21_adv_False_T_adversarial",
        # "peer_3_Room-v21_adv_False_AC_adversarial",
        # "peer_3_Room-v21_adv_False_A_adversarial",
        #"peer_3_Room-v21_adv_False_C_adversarial",


        # "peer_4_HalfCheetah-v4_adv_True_ACT_peer_noise_sample_True__idiot",
        # "peer_2_Room-v21_adv_False__peer_noise_sample_True__mel_10000early_learning",
        #"peer_2_Room-v21_adv_False__peer_noise_sample_True__mel_0early_learning",
        # "peer_4_Room-v21_adv_False__peer_noise_sample_True__mel_10000early_learning",
        # "peer_4_Room-v21_adv_False__peer_noise_sample_True__mel_0early_learning",
        # "peer_2_Room-v27_adv_False__peer_noise_sample_True__mel_10000early_learning",
        # "peer_2_Room-v27_adv_False__peer_noise_sample_True__mel_0early_learning",
        # "peer_4_Room-v27_adv_False__peer_noise_sample_True__mel_10000early_learning",
        # "peer_4_Room-v27_adv_False__peer_noise_sample_True__mel_0early_learning",
        # "peer_1_Room-v21_adv_False__peer_noise_sample_True__lr_lambda_x:_5e-4",
        # "peer_1_Room-v27_adv_False__peer_noise_sample_True__lr_lambda_x:_5e-4",
        # "peer_4_Ant-v4_adv_False__peer_noise_sample_True__early_learning",
        # "peer_4_HalfCheetah-v4_adv_False__peer_noise_sample_True__early_learning",
        # "peer_4_Walker2d-v4_adv_False__peer_noise_sample_True__early_learning",
        # "peer_4_Hopper-v4_adv_False__peer_noise_sample_True__early_learning"
    ]
    wandb_entity_name = 'jgu-wandb'
    wandb_project = 'peer-learning'
    upload_identifier = ""
    Path_to_experiments = Path(
        "/home/jbrugger/PycharmProjects/decentralized-peer-learning/Experiments")

    for setup in directory:
        print(f"\n\n\n-------------------{setup}")
        upload_experiment(Path_to_experiments, setup, upload_identifier,
                          wandb_entity_name, wandb_project)


def upload_experiment(Path_to_experiments, setup, upload_identifier,
                      wandb_entity_name, wandb_project):
    path_to_runs = Path_to_experiments / setup
    list_of_runs = glob.glob(
        f"{path_to_runs}/*")  # * means all if need specific format then *.csv
    runs = get_all_results(list_of_runs)
    runs.sort(reverse=True)
    # runs = get_run_with_most_results(list_of_runs)
    for run in runs:
        try:
            path_to_run_to_upload = path_to_runs / run / 'wandb'
            id = f"{setup}__{run}{upload_identifier}"
            id = id.replace(':', '')
            path_to_run_to_upload = get_path_to_wandb_file(
                path_to_run_to_upload)
            if len(id) >= 120:
                id = id.replace('learning_rate', 'lr')
                id = id.replace('temperature_decay', 'td')
            command = f"wandb sync -e {wandb_entity_name} " \
                      f"-p {wandb_project}" \
                      f" --id {id}" \
                      f" {path_to_run_to_upload} --no-include-synced"
            now = datetime.datetime.now()
            print(f"Current date and time: {now.strftime('%H:%M:%S')}")
            print(command)

            os.system(command)
            try:
                output = check_output(command.split(), stderr=STDOUT, timeout=120)
                print(output)
            except(TimeoutExpired):
                pass

            # pass
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
