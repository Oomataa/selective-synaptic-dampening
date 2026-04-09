#!/bin/bash
#SBATCH --job-name=std_p20_raw_s
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/logs/std_pins20_raw_ssd_%j.out
#SBATCH --error=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/logs/std_pins20_raw_ssd_%j.err

set -euo pipefail

REPO_ROOT=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening
SRC_DIR=${REPO_ROOT}/src
CONDA_ENV=ssd
WEIGHT_PATH=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/src/checkpoint/ResNet18/Wednesday_01_April_2026_13h_30m_58s/ResNet18-PinsFaceRecognition-178-best.pth
DATA_ROOT=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/src/105_classes_pins_dataset

mkdir -p "${REPO_ROOT}/logs"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export WANDB_MODE=disabled
export WANDB_SILENT=true

cd "${SRC_DIR}"
python forget_full_class_main.py -net ResNet18 -dataset PinsFaceRecognition -classes 105 -gpu -method ssd_tuning -forget_class 20 -weight_path "${WEIGHT_PATH}" -seed 1 -data_root "${DATA_ROOT}"
