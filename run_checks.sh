#!/bin/bash

# Load required modules and activate env
module load CBI
unset CONDA_EXE
module load miniforge3/24.11.0-0
conda activate env_sleep_py310

# Define log file and screen session name and command
LOGFILE="/wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/tests/checks_mgb_logfile.txt"
SESSION_NAME="check_events_mgb"
COMMAND="python /wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/tests/check_event_files.py"

# Start screen in detached mode with logging
screen -dmL -Logfile "$LOGFILE" -S "$SESSION_NAME" bash -c "$COMMAND"
echo "Screen session '$SESSION_NAME' started in detached mode."