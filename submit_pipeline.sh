#!/bin/bash
#$ -S /bin/bash       # the shell language when run via the job scheduler
#$ -cwd               # job should run in the current working directory
#$ -j y               # cleaner log
#$ -N vb_shhs1
#$ -o log/vb_shhs1.o$JOB_ID
#$ -l h_rt=72:00:00
#$ -pe smp 16
#$ -l mem_free=12G 
#$ -l h_vmem=24G

# Activate environment
module load CBI
module load miniforge3/24.11.0-0
module load matlab
conda activate env_sleep_py310

# Run your pipeline
python /wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/run_pipeline.py \
    --config /wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/config/shhs_ses-1_config.yaml
