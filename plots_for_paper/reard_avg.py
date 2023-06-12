import pandas as pd
from pathlib import Path
import numpy as np


def make_full_table(data, path=Path("tex_for_paper/table.tex"), keys_wanted=None,
                    tasks_wanted=None, cells=None):
    if keys_wanted is None:
        keys_wanted = list(data.values())[0].keys()
    if tasks_wanted is None:
        tasks_wanted = data.keys()

    with open(path, mode="w") as file:
        file.write(f"\\begin{{table}}\n"
                   f"{' '*4}\\caption{{Comparison of learning speed and "
                   f"final performance expressed as average reward over time"
                   f"($\\pm$ the standard deviation in percent). Values "
                   f"within 2 percent of the maximum are printed bold.}}\n"
                   f"{' '*4}\\begin{{tabularx}}{{\\columnwidth}}{{|c|")
        for i in range(len(list(data.values())[0])):
            file.write("Y|")
        file.write(f"}}\n{' '*4}\\hline\n")
        # header
        max_task_len = max([len(task_) for task_ in tasks_wanted])
        file.write(f"{' '*(4+max_task_len)}")
        for i, k in enumerate(keys_wanted):
            length = max((len(k), len("\\textbf{XX.X}")))
            file.write(" & ")
            file.write(f"{k:{length}s}")
        file.write(f" \\\\\n{' '*4}\\hline\n")
        # data
        for t in tasks_wanted:
            file.write(f"{' '*4}{t}{' ' * (max_task_len - len(t))}")
            for i, k in enumerate(keys_wanted):
                length = max((len(k), len("\\textbf{XX.X}")))
                file.write(" & ")
                if k not in data[t].keys():
                    file.write(f"{'-':{length}s}")
                elif data[t][k] * 1.02 >= max([data[t][kk]] for kk in set(
                                           keys_wanted
                                       ).intersection(data[t].keys()))[0]:
                    file.write(f"\\"
                               f"{f'textbf{{{cells[t][k]}}}':{length-1}s}")
                else:
                    file.write(f"{cells[t][k]:{length}s}")
            file.write(" \\\\\n")
        file.write(f"{' '*4}\\hline\n"
                   f"{' '*4}\\end{{tabularx}}\n"
                   f"{' '*4}\\label{{tab:comparison}}\n"
                   f"\\end{{table}}")


def make_latex_table(data, path=Path("tex_for_paper/table.tex")):
    with open(path, mode="w") as file:
        file.write(f"\\begin{{table}}\n"
                   f"{' '*4}\\caption{{Comparison of the learning speed and "
                   f"final performance in terms of the average reward during "
                   f"training as a more interpretable equivalent of the are "
                   f"under the learning curve. Values within 2 percent of the "
                   f"maximum are printed bold.}}\n"
                   f"{' '*4}\\begin{{tabularx}}{{\\columnwidth}}{{|c|")
        for i in range(len(data)-1):
            file.write("Y|")
        file.write(f"}}\n{' '*4}\\hline\n")
        # header
        for i, k in enumerate(data.keys()):
            file.write(f"{' '*4}")
            if i != 0:
                file.write("& ")
            file.write(f"{k}")
        file.write(f"{' '*4}\\\\\n"
                   f"{' '*4}\\hline\n")
        # data
        for i, k in enumerate(data.keys()):
            file.write(f"{' '*4}")
            if i != 0:
                file.write("& ")
            if data[k] == max(data.values()):
                file.write(f"\\textbf{{{data[k]:{len(k)}.2f}}}")
            else:
                file.write(f"{data[k]:{len(k)}.2f}")
        file.write(f"{' '*4}\\\\\n"
                   f"{' '*4}\\hline\n"
                   f"{' '*4}\\end{{tabularx}}\n"
                   f"{' '*4}\\label{{tab:comparison}}\n"
                   f"\\end{{table}}")


if __name__ == '__main__':
    tasks = [
        "HalfCheetah-v4",
         "Walker2d-v4",
         "Ant-v4",
         "Hopper-v4",
         "Room-v21",
         "Room-v27",
    ]
    results_wanted = [
        # "Peer Learning",
        "Peer Learning with Advantage",
        "Single Agent",
        "Early Advising",
        "Random Advice",
        "LeCTR"
    ]
    results = {}
    cells = {}
    for task in tasks:
        csv = pd.read_csv(Path(f"data/{task}.csv"))
        result = {}
        num = {}
        mean_of_peers = {}
        # for key in csv.keys():
        #     if "MIN" in key or "MAX" in key or "step" in key:
        #         csv.pop(key)
        #     else:
        #         # exp = key.split(" - ")[0]
        #         exp = key[key.find("(")+1:key.find(")")]
        #         if exp in result:
        #             result[exp] += [csv[key]]
        #             # num[exp] += 1
        #         else:
        #             # result[exp] = csv[key]
        #             # num[exp] = 1
        #             result[exp] = [csv[key]]

        # average peers in runs
        for key in csv.keys():
            if "MIN" in key or "MAX" in key or "step" in key:
                csv.pop(key)
            else:
                k_ = key.split(" - ")[0]
                if k_ in mean_of_peers:
                    mean_of_peers[k_] += csv[key]
                    num[k_] += 1
                else:
                    mean_of_peers[k_] = csv[key]
                    num[k_] = 1
        mean_speed_of_seed = {}
        for key in mean_of_peers.keys():
            mean_of_peers[key] /= num[key]
            mean_speed_of_seed[key] = np.nanmean(mean_of_peers[key])

        # average over seeds
        for key in mean_speed_of_seed.keys():
            exp = key[key.find("(")+1:key.find(")")]
            if exp in result:
                result[exp] += [mean_speed_of_seed[key]]
            else:
                result[exp] = [mean_speed_of_seed[key]]

        mean_ = {}
        std = {}
        count = {}
        cell = {}
        for key in result.keys():
            # result[key] /= num[key]
            # result[key] = np.nanmean(result[key])
            # result[key] = np.nanmean(result[key], axis=1)
            mean_[key] = np.nanmean(result[key])
            std[key] = np.nanstd(result[key])
            count[key] = len(result[key])
            cell[key] = f"{mean_[key]:.0f} $\\pm$" \
                        f" {100*std[key]/mean_[key]:.0f}\%"
        # results[task] = result
        results[task] = mean_
        cells[task] = cell
        print(result)
    # for task in tasks:
    #     make_latex_table(results[task], Path(f"./{task}.tex"))
    make_full_table(results, keys_wanted=results_wanted, cells=cells)
