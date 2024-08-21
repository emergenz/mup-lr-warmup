#!/bin/bash

# List of warmup steps to test
warmup_steps_list=(800 1600 3200 6400)

# Base script to run
base_script="main.py"

# Temporary file to store modified scripts
temp_script="temp_script.py"

for warmup_steps in "${warmup_steps_list[@]}"
do
  # Modify the warmup steps in the Python script
  sed "s/'warmup_steps': [0-9]\+/'warmup_steps': $warmup_steps/" $base_script > $temp_script

  # Run the modified script
  python3 $temp_script

  # Clean up temporary script file
  rm $temp_script
done
