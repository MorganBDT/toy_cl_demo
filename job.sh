#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-01:30                         # Runtime in D-HH:MM format
#SBATCH --mem=4000M                         # Memory total in MB (for all cores)
#SBATCH -p gpu                             # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH -o ./results/o2_results_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./results/o2_errors_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=FAIL

module load conda2/4.2.13
module load gcc/6.2.0
module load cuda/10.2

source activate /home/mbt10/.conda/envs/ml1

python cl_fail_net_cifar.py
