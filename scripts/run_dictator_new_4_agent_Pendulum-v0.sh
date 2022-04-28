#!/bin/bash

#SBATCH --job-name=08_04_22_dictator_4_agent_Pendulum-v0      # Job name
#SBATCH -p smp                 # The partition your job should run #in. devel smp parallel
#SBATCH --account=m2_datamining   # Specify allocation to charge against m$
#SBATCH --time=4:00:00           # Run time (hh_mm_ss) - 30 seconds
#SBATCH --tasks=1                 # Total number of tasks (=cores if CPU i$
#SBATCH --nodes=1                 # The number of nodes you need
#SBATCH --cpus-per-task=10         # Total number of cores for the single task
#SBATCH --mem=10G                  # The amount of RAM requested

#SBATCH -o \%x_\%j_profile.out # Specify stdout output file where \%j e$
#SBATCH -C anyarch
#SBATCH -o output_08_04_22/\%x_\%j.out
#SBATCH -e output_08_04_22/\%x_\%j.err

#SBATCH --mail-user=jbrugge@students.uni-mainz.de
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END


module purge
module load devel/PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)
export http_proxy=http://webproxy.zdv.uni-mainz.de:8888
export https_proxy=https://webproxy.zdv.uni-mainz.de:8888
source virt/bin/activate
wandb offline
srun python run_dicator_new.py  --save-name dictator_new_4_agent_Pendulum-v0_08_04_22  --env Pendulum-v0  --agent-count 4 \
 --eval-interval 500 --batch-size 256 --buffer-size 1_000_000 --steps 20_000 --buffer-start-size 100 --learning_rate 1e-3 
