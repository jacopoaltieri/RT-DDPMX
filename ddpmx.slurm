#!/bin/bash

#SBATCH --job-name=ddpmx  
#SBATCH --error=ddpmx-%j.err
#SBATCH --output=ddpmx-%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load anaconda/3.23.07
module load cuda/11.8
module load cudnn/8.9

source activate venv

python ~/ddpmx/code/main.py
