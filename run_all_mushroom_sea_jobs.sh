#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening"
LOG_DIR="${REPO_ROOT}/logs"
STD_WEIGHT="/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/src/checkpoint/ResNet18/Friday_27_March_2026_14h_49m_43s/ResNet18-Cifar100-195-best.pth"
FN_WEIGHT="/home/ikhianosen.ehizokhal/code/SSD/selective-synaptic-dampening/src/checkpoint/ResNet18_FN/Sunday_29_March_2026_20h_42m_48s/ResNet18_FN-Cifar100-200-best.pth"

mkdir -p "${LOG_DIR}"

make_job() {
  local class_name="$1"
  local model_name="$2"
  local method_name="$3"
  local weight_path="$4"
  local job_tag="$5"
  local job_file="${REPO_ROOT}/run_${job_tag}.slurm"

  cat > "${job_file}" <<EOS
#!/bin/bash
#SBATCH --job-name=${job_tag}
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/${job_tag}_%j.out
#SBATCH --error=logs/${job_tag}_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
SRC_DIR="\${REPO_ROOT}/src"
LOG_DIR="\${REPO_ROOT}/logs"
WEIGHT_PATH="${weight_path}"
CONDA_ENV="\${CONDA_ENV:-ssd}"
SEED="\${SEED:-1}"

mkdir -p "\${LOG_DIR}"

source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "\${CONDA_ENV}"
export WANDB_MODE=disabled
export WANDB_SILENT=true

cd "\${SRC_DIR}"

echo "[INFO] Job started at \$(date)"
echo "[INFO] Running on host \$(hostname)"
echo "[INFO] Using checkpoint: \${WEIGHT_PATH}"
echo "[INFO] Seed: \${SEED}"
echo "[INFO] Model: ${model_name}"
echo "[INFO] Forget class: ${class_name}"
echo "[INFO] Method: ${method_name}"

nvidia-smi || true

python forget_full_class_main.py \
  -net ${model_name} \
  -dataset Cifar100 \
  -classes 100 \
  -gpu \
  -method ${method_name} \
  -forget_class ${class_name} \
  -weight_path "\${WEIGHT_PATH}" \
  -seed "\${SEED}"

echo "[INFO] Job finished at \$(date)"
EOS

  chmod +x "${job_file}"
}

make_job mushroom ResNet18 baseline   "${STD_WEIGHT}" std_mushroom_baseline
make_job mushroom ResNet18 ssd_tuning "${STD_WEIGHT}" std_mushroom_ssd
make_job mushroom ResNet18_FN baseline   "${FN_WEIGHT}" fn_mushroom_baseline
make_job mushroom ResNet18_FN ssd_tuning "${FN_WEIGHT}" fn_mushroom_ssd

make_job sea ResNet18 baseline   "${STD_WEIGHT}" std_sea_baseline
make_job sea ResNet18 ssd_tuning "${STD_WEIGHT}" std_sea_ssd
make_job sea ResNet18_FN baseline   "${FN_WEIGHT}" fn_sea_baseline
make_job sea ResNet18_FN ssd_tuning "${FN_WEIGHT}" fn_sea_ssd

for script in \
  run_std_mushroom_baseline.slurm \
  run_std_mushroom_ssd.slurm \
  run_fn_mushroom_baseline.slurm \
  run_fn_mushroom_ssd.slurm \
  run_std_sea_baseline.slurm \
  run_std_sea_ssd.slurm \
  run_fn_sea_baseline.slurm \
  run_fn_sea_ssd.slurm
  do
    echo "Submitting ${script}"
    sbatch "${REPO_ROOT}/${script}"
  done
