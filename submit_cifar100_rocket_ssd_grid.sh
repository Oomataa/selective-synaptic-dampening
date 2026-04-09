#!/bin/bash
set -euo pipefail

REPO_ROOT=/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening
FORGET_CLASS=${FORGET_CLASS:-rocket}
NET=${NET:-ResNet18}
SEED=${SEED:-1}
WEIGHT_PATH=${WEIGHT_PATH:-/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/src/checkpoint/ResNet18/Friday_27_March_2026_14h_49m_43s/ResNet18-Cifar100-195-best.pth}
DAMPENING_VALUES=${DAMPENING_VALUES:-"0.5 1 2"}
SELECTION_VALUES=${SELECTION_VALUES:-"5 10 20"}
VALIDATION_MODE=${VALIDATION_MODE:-split}
VALIDATION_SPLIT_RATIO=${VALIDATION_SPLIT_RATIO:-0.1}
SPLIT_SEED=${SPLIT_SEED:-${SEED}}

cd "${REPO_ROOT}"
for damp in ${DAMPENING_VALUES}; do
  for sel in ${SELECTION_VALUES}; do
    job_name="valsplit_${FORGET_CLASS}_${NET}_${damp}_${sel}"
    sbatch \
      --job-name="${job_name}" \
      --export=ALL,NET="${NET}",FORGET_CLASS="${FORGET_CLASS}",WEIGHT_PATH="${WEIGHT_PATH}",SEED="${SEED}",DAMPENING_CONSTANT="${damp}",SELECTION_WEIGHTING="${sel}",VALIDATION_MODE="${VALIDATION_MODE}",VALIDATION_SPLIT_RATIO="${VALIDATION_SPLIT_RATIO}",SPLIT_SEED="${SPLIT_SEED}",JOB_TAG="${job_name}" \
      run_cifar100_ssd_tune.slurm
  done
done
