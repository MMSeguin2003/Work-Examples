#!/bin/bash

# SBATCH options

#SBATCH --job-name=na         # job name for queue (optional)
#SBATCH --partition=###       # partition (optional, default=low)
#SBATCH --error=ex.err        # file for stderr (optional)
#SBATCH --output=ex.out       # file for stdout (optional)
#SBATCH --time=01:00:00       # max runtime of job hours:minutes:seconds
#SBATCH --ntasks-per-node=16  # use 16 CPU cores
#SBATCH --wait                # don't continue until job finished

# Command(s) to run

# Run python script
python script2.py
