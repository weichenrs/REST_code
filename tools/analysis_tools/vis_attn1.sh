#!/bin/bash
#SBATCH --partition=hpxg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

python vis_attn1.py