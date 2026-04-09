#!/bin/bash
#SBATCH --job-name=ssd_rn18_pretrain
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/pretrain_resnet18_%j.out
#SBATCH --error=logs/pretrain_resnet18_%j.err

set -euo pipefail

REPO_ROOT="/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening"
SRC_DIR="${REPO_ROOT}/src"
LOG_DIR="${REPO_ROOT}/logs"

# Override these at submission time if needed, for example:
# sbatch --export=ALL,DATASET=Cifar20,N_CLASSES=20 run_pretrain_resnet18.slurm
CONDA_ENV="${CONDA_ENV:-ssd}"
DATASET="${DATASET:-Cifar100}"
N_CLASSES="${N_CLASSES:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
USE_GPU="${USE_GPU:-1}"

mkdir -p "${LOG_DIR}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${SRC_DIR}"

echo "[INFO] Job started at $(date)"
echo "[INFO] Running on host $(hostname)"
echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Classes: ${N_CLASSES}"
echo "[INFO] Batch size: ${BATCH_SIZE}"
echo "[INFO] Learning rate: ${LEARNING_RATE}"

nvidia-smi || true

CMD=(
    python pretrain_model.py
    -net ResNet18
    -dataset "${DATASET}"
    -classes "${N_CLASSES}"
    -b "${BATCH_SIZE}"
    -lr "${LEARNING_RATE}"
)

if [ "${USE_GPU}" = "1" ]; then
    CMD+=(-gpu)
fi

echo "[INFO] Command: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Job finished at $(date)"
echo "[INFO] To find the saved checkpoint, run:"
echo "find ${SRC_DIR}/checkpoint/ResNet18 -name '*${DATASET}*best.pth'"
