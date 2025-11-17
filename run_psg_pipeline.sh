#!/bin/bash

# Load required modules and activate env
module load CBI
unset CONDA_EXE
module load miniforge3/24.11.0-0
module load matlab
conda activate env_sleep_py310

# Define log file and screen session name adn command
LOGFILE="/wynton/home/leng/alice-albrecht/PSG_Pipeline/bidmc_logfile.txt"
SESSION_NAME="bidmc"
COMMAND="python /wynton/home/leng/alice-albrecht/PSG_Pipeline/run_pipeline.py --config /wynton/home/leng/alice-albrecht/PSG_Pipeline/config/hsp_bidmc_config.yaml"

# Start screen in detached mode with logging
screen -dmL -Logfifor noy le "$LOGFILE" -S "$SESSION_NAME" bash -c "$COMMAND"
echo "Screen session '$SESSION_NAME' started in detached mode."