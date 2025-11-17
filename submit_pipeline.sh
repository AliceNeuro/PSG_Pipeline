#!/bin/bash
#$ -S /bin/bash       # the shell language when run via the job scheduler
#$ -cwd               # job should run in the current working directory
#$ -j y               # cleaner log
#$ -N bidmc_features
#$ -o log/bidmc_features.o$JOB_ID
#$ -l h_rt=48:00:00
#$ -pe smp 8
#$ -l mem_free=2G 

# Load environment
module load CBI
unset CONDA_EXE
module load miniforge3/24.11.0-0
module load matlab
conda activate env_sleep_py310

# Run your pipeline
python /wynton/home/leng/alice-albrecht/PSG_Pipeline/run_pipeline.py \
    --config /wynton/home/leng/alice-albrecht/PSG_Pipeline/config/hsp_bidmc_config.yaml
