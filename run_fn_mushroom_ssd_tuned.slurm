#!/bin/bash
#SBATCH --job-name=fn_mushroom_tuned
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/logs/fn_mushroom_ssd_tuned_%j.out
#SBATCH --error=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/logs/fn_mushroom_ssd_tuned_%j.err

set -euo pipefail

REPO_ROOT=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening
SRC_DIR=${REPO_ROOT}/src
CONDA_ENV=ssd
WEIGHT_PATH=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/src/checkpoint/ResNet18_FN/Sunday_29_March_2026_20h_42m_48s/ResNet18_FN-Cifar100-200-best.pth

mkdir -p "${REPO_ROOT}/logs"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export WANDB_MODE=disabled
export WANDB_SILENT=true

cd "${SRC_DIR}"
python forget_full_class_main.py   -net ResNet18_FN   -dataset Cifar100   -classes 100   -gpu   -method ssd_tuning   -forget_class mushroom   -weight_path "${WEIGHT_PATH}"   -seed 1   -dampening_constant 0.5   -selection_weighting 5
