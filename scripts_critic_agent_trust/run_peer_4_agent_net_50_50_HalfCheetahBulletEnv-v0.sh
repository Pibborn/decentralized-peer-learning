#!/bin/bash

#SBATCH --job-name=peer_4_agent_net_50_50_HalfCheetahBulletEnv-v0      # Job name
#SBATCH -p smp                 # The partition your job should run #in. devel smp parallel
#SBATCH --account=m2_datamining   # Specify allocation to charge against m$
#SBATCH --time=96:00:00          # Run time (hh_mm_ss) - 30 seconds
#SBATCH --tasks=1                 # Total number of tasks (=cores if CPU i$
#SBATCH --nodes=1                 # The number of nodes you need
#SBATCH --cpus-per-task=10         # Total number of cores for the single task
#SBATCH --mem=10G                  # The amount of RAM requested

#SBATCH -o \%x_\%j_profile.out # Specify stdout output file where \%j e$
#SBATCH -C anyarch
#SBATCH -o output_11_05_22/\%x_\%j.out
#SBATCH -e output_11_05_22/\%x_\%j.err

#SBATCH --mail-user=jbrugge@students.uni-mainz.de
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

module purge
module load devel/SWIG/4.0.1-foss-2019b-Python-3.7.4
module load devel/PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)
export http_proxy=http://webproxy.zdv.uni-mainz.de:8888
export https_proxy=https://webproxy.zdv.uni-mainz.de:8888
source virt/bin/activate
wandb offline
srun python run_peer.py --save-name $SLURM_JOB_NAME \
  --job_id $SLURM_JOB_ID --env HalfCheetahBulletEnv-v0 --agent-count 4 --batch-size 256 --buffer-size 300_000 \
  --steps 1_000_000 --buffer-start-size 10_000 --learning_rate 7.3e-4 --gamma 0.98 --gradient_steps 8 --tau 0.02 --train_freq 8 \
  --seed $SLURM_JOB_ID --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise False\
  --net-arch 50  50
