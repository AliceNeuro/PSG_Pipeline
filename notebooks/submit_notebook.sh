#!/bin/bash
#$ -S /bin/bash       # the shell language when run via the job scheduler
#$ -cwd               # job should run in the current working directory
#$ -j y               # cleaner log
#$ -N combined_bidmc
#$ -o combined_bidmc.o$JOB_ID
#$ -l h_rt=2:00:00
#$ -l mem_free=30G 

# Load environment
module load CBI
unset CONDA_EXE
module load miniforge3/24.11.0-0
conda activate env_sleep_py310

# Run your pipeline
python /wynton/home/leng/alice-albrecht/PSG_Pipeline/notebooks/combined_extracted_features.py
