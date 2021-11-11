
import os 
import numpy as np
import matplotlib.pyplot as plt
import json


current_dir = os.path.dirname(__file__)

save_path = os.path.join(current_dir,"result")

experiment1 = [
    "dictator",
    "dictator1",
    "dictator2",
    "dictator3",
    "dictator4",
    ]

experiment2 = [
    "main",
    "main1",
    "main2",
    "main3",
    "main4",
]

followed = []
y1 = []
std_1 = []
y2 = []
std_2 = []
for j in range(len(experiment1)):

    save_name = experiment1[j]
    experiment_folder = os.path.join(current_dir,"Experiments",save_name)#f"{current_dir}\\Experiments\\{save_name}"
    result_folder = os.path.join(experiment_folder,"results.json")#f"{experiment_folder}\\results.json"

    with open(result_folder, "r") as file:
        results = json.load(file)
        test_rewards = results["rewards"]
    
    result = test_rewards[3]
    y1.append(result)

    if (j < len(experiment2)):
        save_name = experiment2[j]
        experiment_folder = os.path.join(current_dir,"Experiments",save_name)#f"{current_dir}\\Experiments\\{save_name}"
        result_folder = os.path.join(experiment_folder,"results.json")#f"{experiment_folder}\\results.json"

        with open(result_folder, "r") as file:
            results = json.load(file)
            test_rewards = results["rewards"]
            
        result = np.mean(test_rewards,axis=0)
        y2.append(result)
        std_2.append(np.std(test_rewards,axis=0))

min_length = np.min([len(x) for x in y1])
y1_err = np.std([x[:min_length] for x in y1],axis=0)
y1 = np.mean([x[:min_length] for x in y1],axis=0)
std_1_err = np.std([x[:min_length] for x in std_1],axis=0)
std_1 = np.mean([x[:min_length] for x in std_1],axis=0)
x1 = np.arange(len(y1))

min_length = np.min([len(x) for x in y2])
y2_err = np.std([x[:min_length] for x in y2],axis=0)
y2 = np.mean([x[:min_length] for x in y2],axis=0)
std_2_err = np.std([x[:min_length] for x in std_2],axis=0)
std_2 = np.mean([x[:min_length] for x in std_2],axis=0)
x2 = np.arange(len(y2))

plt.errorbar(x1,y1,label="dictator")
plt.fill_between(x1, y1-y1_err, y1 + y1_err, alpha=0.2)

plt.errorbar(x2,y2,label="peer")
plt.fill_between(x2,y2-y2_err,y2+y2_err, alpha=0.2)
plt.legend()

plt.ylabel("reward")
plt.xlabel("epoch")

plt.savefig(save_path)
plt.show()

plt.errorbar(x1,std_1,label="dictator")
plt.fill_between(x1, std_1-std_1_err, std_1 + std_1_err, alpha=0.2)

plt.errorbar(x2,std_2,label="peer")
plt.fill_between(x2, std_2-std_2_err, std_2+std_2_err, alpha=0.2)
plt.legend()

plt.ylabel("standard deviation")
plt.xlabel("epoch")

plt.savefig(save_path+"_std")
plt.show()
