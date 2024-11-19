#!/bin/bash

# SBATCH options

#SBATCH --job-name=na        # job name for queue (optional)
#SBATCH --partition=###      # partition (optional, default=low)
#SBATCH --error=ex.err       # file for stderr (optional)
#SBATCH --output=ex.out      # file for stdout (optional)
#SBATCH --time=00:10:00      # max runtime of job hours:minutes:seconds
#SBATCH --ntasks-per-node=4  # use 4 CPU cores
#SBATCH --mem-per-cpu=5G     # limit memory per cpu to 5GB
#SBATCH --wait               # don't continue until job finished

# Command(s) to run

# Make my subdirectory in tmp
mkdir /tmp/subdir
# Copy all files in given directory to tmp
scp /datadir/* /tmp/subdir
# Run python script
python script1.py
# Remove files from tmp
rm -rf /tmp/subdir
