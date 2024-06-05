#!/bin/bash

#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=interactive_nnunetv2
#SBATCH --output=/home/<username>/logs/slurm-%j.out

# First runs install of the latest verison available on our cluster, feel free to adjust the path to your own clone on the cluster
# Then launches jupyter lab

srun \
  --container-mounts=/data/pathology:/data/pathology \
  --container-image="doduo2.umcn.nl#nnunet_for_pathology/sol2:latest" \
  /bin/bash -c " \
    pip3 install --no-use-pep517 -e /data/pathology/projects/nnUNet_v2>    && jupyter lab --ip=0.0.0.0 --port=<port]> --no-browser --notebook-dir=~ \
  "
