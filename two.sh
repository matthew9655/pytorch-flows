#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=2
#SBATCH --time=22:00:00
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --output=slurminfo0

export OMP_NUM_THREADS=1
module load NiaEnv/2019b python/3.8.5
module load NiaEnv/2019b gnu-parallel
source ~/.virtualenvs/qmc/bin/activate

parallel --joblog slurm_parallel/slurm-$SLURM_JOBID.log --wd $PWD python3 run_two.py {1} ::: $(seq 1 80)
