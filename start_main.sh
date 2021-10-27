#!/bin/bash

#SBATCH --job-name=main       # Job name
#SBATCH -p smp                 # The partition your job should run #in. devel smp parallel
#SBATCH --account=m2_datamining   # Specify allocation to charge against m$
#SBATCH --time=120:00:00           # Run time (hh_mm_ss) - 30 seconds
#SBATCH --tasks=1                 # Total number of tasks (=cores if CPU i$
#SBATCH --nodes=1                 # The number of nodes you need
#SBATCH --cpus-per-task=12         # Total number of cores for the single task
#SBATCH --mem=10G                  # The amount of RAM requested

#SBATCH -o \%x_\%j_profile.out # Specify stdout output file where \%j e$

#SBATCH -o output/\%x_\%j.out
#SBATCH -e output/\%x_\%j.err

#SBATCH --mail-user=apuhl@students.uni-mainz.de
#SBATCH --mail-type=FAIL

module purge
module load devel/PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
export PYTHONPATH=$PYTHONPATH:$(pwd)
source virt/bin/activate

srun python main.py --save-name="main" --seed=100 --agent-count=4 --gpu=-1
