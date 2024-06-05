#!/bin/bash

dataset=$1
fold=$2
trainer=$3

# Get the original current directory
ORIGINAL_DIR=$(pwd)

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script's directory so we can use ralative paths
cd "$DIR"

# Install the required library
pip3 install --no-use-pep517 -e .. # in case you have problems with nnUNet's (updated) installation, try adding --no-use-pep517 before -e
git config --global --add safe.directory ..

export WANDB_API_KEY=yourapikeyhere

# setting these paths here, feel free to comment out and specigy them somewhere else
export nnUNet_raw="/your/folder/here/nnUNet_raw"
export nnUNet_preprocessed="/your/folder/here/nnUNet_preprocessed"
export nnUNet_results="/your/folder/here/nnUNet_results"

# Check if WANDB_API_KEY is defined
if [ -n "${WANDB_API_KEY}" ]; then
    echo "USING WANDB API KEY"
    wandb login ${WANDB_API_KEY}
else
    echo "WANDB_API_KEY is not defined. WandB login skipped."
fi

# Run the Python script
echo ---------------------------------
echo INSTALLS DONE, START PREPROCESSING
echo ---------------------------------
python3 ../nnunetv2/experiment_planning/experiment_planners/pathology_experiment_planner.py "$dataset"


echo ---------------------------------
echo PREPROCESSING DONE, START TRAINING
echo ---------------------------------
python3 ../nnunetv2/run/run_training_pathology.py "$dataset" "$fold" "$trainer"

echo ---------------------------------
echo TRAINING DONE
echo ---------------------------------
echo TOTALLY DONE

# Return to the original directory
cd "$ORIGINAL_DIR"