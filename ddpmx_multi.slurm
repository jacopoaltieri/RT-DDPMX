#!/bin/bash

#SBATCH --job-name=ddpmx_ddp  
#SBATCH --error=ddpmx-%j.err
#SBATCH --output=ddpmx-%j.out
#SBATCH --ntasks=16                 # Number of tasks
#SBATCH --gres=gpu:16               # Request 16 GPUs
#SBATCH --mem=64000                # Memory per node

module load anaconda/3.23.07
module load cuda/11.8
module load cudnn/8.9

source activate venv

# Launch the distributed training using torchrun with 8 processes (one per GPU)
torchrun --nproc_per_node=16 ~/ddpmx/code/main_multi.py
