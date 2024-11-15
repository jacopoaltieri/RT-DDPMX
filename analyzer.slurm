#!/bin/bash

#SBATCH --job-name=ddpmx  
#SBATCH --error=ddpmx_an-%j.err
#SBATCH --output=ddpmx_an-%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --partition=shortrun

module load anaconda/3.23.07
module load cuda/11.8
module load cudnn/8.9

source activate venv

python ~/ddpmx/code/analyzer.py
