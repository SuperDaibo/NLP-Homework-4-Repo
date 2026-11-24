#!/bin/bash
#SBATCH --job-name=a4-t5-ec-test-final
#SBATCH --account=torch_pr_932_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/a4-t5-ec-test-final-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

apptainer exec --nv --bind /scratch \
  --overlay "$PWD/a4-python.ext3:ro" \
  /share/apps/images/ubuntu-22.04.4.sif \
  /ext3/venv/bin/python -u - <<'PY'
import torch
from transformers import T5ForConditionalGeneration
from load_data import get_dataloader
from train_t5 import test_inference

assert torch.cuda.is_available(), "GPU unavailable"
device = torch.device("cuda")

checkpoint = (
    "checkpoints/scr_experiments/"
    "scratch_continue_lr3e5_24290877/best"
)

model = T5ForConditionalGeneration.from_pretrained(
    checkpoint
).to(device)

loader = get_dataloader(16, "test")

test_inference(
    args=None,
    model=model,
    test_loader=loader,
    model_sql_path="results/t5_ft_experiment_ec_test.sql",
    model_record_path="records/t5_ft_experiment_ec_test.pkl",
)
PY
