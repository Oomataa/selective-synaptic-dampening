#!/bin/bash
#SBATCH --job-name=ssd_tune
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/logs/%x_%j.out
#SBATCH --error=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/logs/%x_%j.err

set -euo pipefail

REPO_ROOT=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening
SRC_DIR=${REPO_ROOT}/src
CONDA_ENV=${CONDA_ENV:-ssd}
NET=${NET:?NET is required}
FORGET_CLASS=${FORGET_CLASS:?FORGET_CLASS is required}
WEIGHT_PATH=${WEIGHT_PATH:?WEIGHT_PATH is required}
SEED=${SEED:-1}
DAMPENING_CONSTANT=${DAMPENING_CONSTANT:?DAMPENING_CONSTANT is required}
SELECTION_WEIGHTING=${SELECTION_WEIGHTING:?SELECTION_WEIGHTING is required}
VALIDATION_MODE=${VALIDATION_MODE:-split}
VALIDATION_SPLIT_RATIO=${VALIDATION_SPLIT_RATIO:-0.1}
SPLIT_SEED=${SPLIT_SEED:-${SEED}}
JOB_TAG=${JOB_TAG:-ssd_tune}

mkdir -p "${REPO_ROOT}/logs"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export WANDB_MODE=disabled
export WANDB_SILENT=true

cd "${SRC_DIR}"
python forget_full_class_main.py \
  -net "${NET}" \
  -dataset Cifar100 \
  -classes 100 \
  -gpu \
  -method ssd_tuning \
  -forget_class "${FORGET_CLASS}" \
  -weight_path "${WEIGHT_PATH}" \
  -seed "${SEED}" \
  -dampening_constant "${DAMPENING_CONSTANT}" \
  -selection_weighting "${SELECTION_WEIGHTING}" \
  -validation_mode "${VALIDATION_MODE}" \
  -validation_split_ratio "${VALIDATION_SPLIT_RATIO}" \
  -split_seed "${SPLIT_SEED}"
