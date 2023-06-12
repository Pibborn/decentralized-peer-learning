import os
import numpy as np

import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt

from reard_avg import make_full_table


def run(args):
    id_of_experiments = {'ACT F S': 'tab:blue',
                         ' AC_ F S': 'tab:orange',
                         'A_T F S': 'tab:green',
                         ' _CT F S': 'tab:red',
                         ' A__ F S': 'tab:brown',
                         ' __T F S': 'tab:pink',
                         # '_C_ X S': 'tab:purple',
                         'ACT A S': 'tab:blue',
                         ' AC_ A S': 'tab:orange',
                         'A_T A S': 'tab:green',
                         ' _CT A S': 'tab:red',
                         ' A__ A S': 'tab:brown',
                         ' __T A S': 'tab:pink',
                         ' Single Agent': 'tab:gray',
                         'follow random': 'tab:olive'
                         }
    result_dict = {}
    experiment_dict = {}

    read_env_data_from_csv(args, experiment_dict)
    for env, df in experiment_dict.items():
        average_experiments_for_one_env(df, env, id_of_experiments,
                                        result_dict)
    mean_of_env = {}
    mean_of_env['global_steps'] = result_dict['Cheetah']['global_step']
    for id in id_of_experiments.keys():
        average_experiments_across_environments(experiment_dict, id,
                                                mean_of_env, result_dict)
    create_plot(args, id_of_experiments, mean_of_env)

    result_dict['Average'] = mean_of_env
    average_reward_over_time = {}
    for k in result_dict.keys():
        average_reward_over_time[k] = {}
        for kk in result_dict[k].keys():
            if "step" in kk:
                continue
            average_reward_over_time[k][kk] = np.nanmean(result_dict[k][kk][
                                                             "mean"])
    make_full_table(average_reward_over_time, Path(
        "tex_for_paper/ablation.tex"))


def create_plot(args, id_of_experiments, mean_of_env):
    fig, ax = plt.subplots()
    for id in id_of_experiments.keys():
        ax.plot(mean_of_env['global_steps'],
                mean_of_env[id]['mean'],
                label=id.replace('_', ''),
                c=id_of_experiments[id],
                linestyle='dashed' if 'A S' in id else 'solid')
    ax.legend()
    save_path = Path(args['path_to_project']) / 'plots_for_paper/plots'
    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(fname=save_path / 'plot.png',
                dpi=500)


def average_experiments_across_environments(experiment_dict, id, mean_of_env,
                                            result_dict):
    mean_of_env[id] = {}
    df_mean = pd.concat([result_dict[env][id]['mean'] for env in
                         experiment_dict.keys()], axis=1)
    mean_above_environments = df_mean.mean(axis=1, numeric_only=True)
    mean_of_env[id]['mean'] = mean_above_environments


def read_env_data_from_csv(args, experiment_dict):
    experiment_dict['Cheetah'] = pd.read_csv(
        Path(args['path_to_project']) / args['path_half_cheetah']
    )
    experiment_dict['Ant'] = pd.read_csv(
        Path(args['path_to_project']) / args['path_ant']
    )
    experiment_dict['Hopper'] = pd.read_csv(
        Path(args['path_to_project']) / args['path_hopper']
    )
    experiment_dict['Walker'] = pd.read_csv(
        Path(args['path_to_project']) / args['path_walker']
    )


def average_experiments_for_one_env(df, env, id_of_experiments, result_dict):
    result_dict[env] = {}
    df_column_names = df.columns.values.tolist()
    # normalize_values_in_one_env(df, df_column_names)
    result_dict[env]['global_step'] = df['global_step']
    max_of_env = 0
    for id in id_of_experiments.keys():
        average_one_experiment_in_one_env(df, df_column_names, env, id,
                                          result_dict)
        if max(result_dict[env][id]['mean']) > max_of_env:
            max_of_env = max(result_dict[env][id]['mean'])
    # normalize
    for id in id_of_experiments.keys():
        result_dict[env][id]['mean'] /= max_of_env


def normalize_values_in_one_env(df, df_column_names):
    max_column_name = get_columns_including_str(
        df_column_names,
        regex=r'.*mean_reward__MAX',
    )
    max_value_columns = df[max_column_name].max()
    max_value = max_value_columns.max()
    mean_reward_columns = get_columns_including_str(
        df_column_names,
        regex=r'.*mean_reward(?!.*__M)',
    )
    df[mean_reward_columns] = df[mean_reward_columns].divide(
        max_value)


def average_one_experiment_in_one_env(df, df_column_names, env, id,
                                      result_dict):
    result_dict[env][id] = {}
    mean, std, step = average_peers_in_one_experiment(
        df=df,
        experiment_id=id,
        df_column_names=df_column_names
    )
    result_dict[env][id]['mean'] = mean
    result_dict[env][id]['std'] = std
    result_dict[env][id]['step'] = step


def average_peers_in_one_experiment(df, df_column_names, experiment_id):
    experiment_columns = get_columns_including_str(
        df_column_names,
        regex=rf'.*{experiment_id}.*mean_reward(?!.*__M)',
    )
    experiment_step = get_columns_including_str(
        df_column_names,
        regex=rf'.*{experiment_id}.*step(?!.*__M)'
    )
    mean = df[experiment_columns].mean(axis=1, numeric_only=True)
    std = df[experiment_columns].std(axis=1, numeric_only=True)
    step = df[experiment_step]
    return mean, std, step


def get_columns_including_str(df_column_names, regex):
    max_column_name = []
    for feature_name in df_column_names:
        if re.match(regex, feature_name):
            max_column_name.append(feature_name)
    return max_column_name


if __name__ == '__main__':
    args = {
        'path_to_project': os.getcwd(),
        'path_half_cheetah': 'data/half_cheetah_150_200_ablation.csv',
        'path_ant': 'data/Ant_150_200_ablation.csv',
        'path_hopper': 'data/hopper_150_200_ablation.csv',
        'path_walker': 'data/Walker_150_200_ablation.csv'
    }
    run(args)
