#!/bin/bash
#$ -S /bin/bash       # the shell language when run via the job scheduler
#$ -cwd               # job should run in the current working directory
#$ -j y               # cleaner log
#$ -N shhs1_vb
#$ -o log/shhs1_vb.o$JOB_ID
#$ -l h_rt=02:00:00
#$ -pe smp 1
#$ -l mem_free=10G 

# Activate environment
module load CBI
module load miniforge3/24.11.0-0
module load matlab
conda activate env_sleep_py310

# Run your pipeline
python /wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/run_pipeline.py \
    --config /wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/config/shhs_ses-1_config.yaml
