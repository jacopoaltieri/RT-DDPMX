#!/bin/bash
#SBATCH --job-name=ddpmx  
#SBATCH --error=ddpmx-%j.err
#SBATCH --output=ddpmx-%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load anaconda/3.23.07 
  

python ~/ddpmx/code/main.py
