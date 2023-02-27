import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt


def run(args):
    id_of_experiments = { 'ACT F S': 'tab:blue',
                         ' AC_ F S': 'tab:orange',
                         'A_T F S': 'tab:green',
                         ' _CT F S':'tab:red',
                         '_C_ F S':'tab:purple',
                         ' A__ F S':'tab:brown',
                         ' __T F S':'tab:pink',
                         ' Single Agent':'tab:gray',
                         'follow random':'tab:olive'
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


def create_plot(args, id_of_experiments, mean_of_env):
    fig, ax = plt.subplots()
    for id in id_of_experiments.keys():
        ax.plot(mean_of_env['global_steps'],
                mean_of_env[id]['mean_of_means'],
                label=id.replace('_',''),
                c=id_of_experiments[id])
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
    mean_above_environments = df_mean.mean(axis=1)
    mean_of_env[id]['mean_of_means'] = mean_above_environments


def read_env_data_from_csv(args, experiment_dict):
    experiment_dict['Cheetah'] = pd.read_csv(
        Path(args['path_to_project']) / args['path_half_cheetah']
    )
    # experiment_dict['Ant'] = pd.read_csv(
    #     Path(args['path_to_project']) / args['path_ant']
    # )
    experiment_dict['Hopper'] = pd.read_csv(
        Path(args['path_to_project']) / args['path_hopper']
    )
    experiment_dict['Walker'] = pd.read_csv(
        Path(args['path_to_project']) / args['path_walker']
    )


def average_experiments_for_one_env(df, env, id_of_experiments, result_dict):
    result_dict[env] = {}
    df_column_names = df.columns.values.tolist()
    normalize_values_in_one_env(df, df_column_names)
    result_dict[env]['global_step'] = df['global_step']
    for id in id_of_experiments.keys():
        average_one_experiment_in_one_env(df, df_column_names, env, id,
                                          result_dict)


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
    mean = df[experiment_columns].mean(axis=1)
    std = df[experiment_columns].std(axis=1)
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
        'path_to_project': '/home/jbrugger/PycharmProjects/decentralized'
                           '-peer-learning',
        'path_half_cheetah':
            'plots_for_paper/data/half_cheetah_150_200_ablation'
            '.csv',
        'path_ant': 'plots_for_paper/data/Ant_150_200_ablation.csv',
        'path_hopper': 'plots_for_paper/data/hopper_150_200_ablation.csv',
        'path_walker': 'plots_for_paper/data/Walker_150_200_ablation.csv'
    }
    run(args)
