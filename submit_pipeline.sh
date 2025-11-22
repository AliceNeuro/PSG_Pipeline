#!/bin/bash
#$ -S /bin/bash       # the shell language when run via the job scheduler
#$ -cwd               # job should run in the current working directory
#$ -j y               # cleaner log
#$ -N mgb_hrv_cpc
#$ -o log/mgb_hrv_cpc.o$JOB_ID
#$ -l h_rt=48:00:00
#$ -pe smp 16
#$ -l mem_free=2G 

# Activate environment
module load CBI
module load miniforge3/24.11.0-0
module load matlab
conda activate env_sleep_py310

# Run your pipeline
python /wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/run_pipeline.py \
    --config /wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/config/hsp_mgb_config.yaml
