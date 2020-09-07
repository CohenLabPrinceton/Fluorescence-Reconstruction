#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jl40@princeton.edu
#SBATCH --output=name_of_output_file.out
#SBATCH --mem=250G

module purge
module load anaconda3
module load cudnn/cuda-10.1
module list

env | grep SLURM

source activate tf-gpu
python ./main.py

