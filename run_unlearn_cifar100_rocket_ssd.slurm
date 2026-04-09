#!/bin/bash
#SBATCH --job-name=ssd_unl_rocket_ssd
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/unlearn_cifar100_rocket_ssd_%j.out
#SBATCH --error=logs/unlearn_cifar100_rocket_ssd_%j.err

set -euo pipefail

REPO_ROOT="/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening"
SRC_DIR="${REPO_ROOT}/src"
LOG_DIR="${REPO_ROOT}/logs"
WEIGHT_PATH="/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/src/checkpoint/ResNet18/Friday_27_March_2026_14h_49m_43s/ResNet18-Cifar100-195-best.pth"
CONDA_ENV="${CONDA_ENV:-ssd}"
SEED="${SEED:-1}"

mkdir -p "${LOG_DIR}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export WANDB_MODE=disabled
export WANDB_SILENT=true

cd "${SRC_DIR}"

echo "[INFO] Job started at $(date)"
echo "[INFO] Running on host $(hostname)"
echo "[INFO] Using checkpoint: ${WEIGHT_PATH}"
echo "[INFO] Seed: ${SEED}"

nvidia-smi || true

python forget_full_class_main.py \
  -net ResNet18 \
  -dataset Cifar100 \
  -classes 100 \
  -gpu \
  -method ssd_tuning \
  -forget_class rocket \
  -weight_path "${WEIGHT_PATH}" \
  -seed "${SEED}"

echo "[INFO] Job finished at $(date)"
