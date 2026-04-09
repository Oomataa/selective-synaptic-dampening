#!/bin/bash
#SBATCH --job-name=ssd_pins_std
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/logs/pretrain_resnet18_pins_%j.out
#SBATCH --error=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/logs/pretrain_resnet18_pins_%j.err

set -euo pipefail

REPO_ROOT=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening
SRC_DIR=${REPO_ROOT}/src
CONDA_ENV=ssd

mkdir -p "${REPO_ROOT}/logs"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${SRC_DIR}"
python pretrain_model.py -net ResNet18 -dataset PinsFaceRecognition -classes 105 -b 64 -lr 0.1 -gpu
