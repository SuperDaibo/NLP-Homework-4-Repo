#!/bin/bash
#SBATCH --job-name=a4-t5-scratch
#SBATCH --account=torch_pr_932_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/a4-t5-scratch-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset A4_START_CHECKPOINT

run="scratch_lr3e4_${SLURM_JOB_ID}"

apptainer exec --nv --bind /scratch \
  --overlay "$PWD/a4-python.ext3:ro" \
  /share/apps/images/ubuntu-22.04.4.sif \
  /ext3/venv/bin/python -u - \
    --learning_rate 3e-4 \
    --batch_size 16 \
    --test_batch_size 16 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --max_n_epochs 30 \
    --patience_epochs 10 \
    --experiment_name "$run" <<'PY'
import os
import torch
from transformers import set_seed
import train_t5 as t

assert torch.cuda.is_available(), "GPU unavailable"
set_seed(42)

args = t.get_args()
assert not args.finetune
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Configuration:", vars(args), flush=True)
print("Training T5 from random weights.", flush=True)

train_loader, dev_loader, _ = t.load_t5_data(
    args.batch_size, args.test_batch_size
)
model = t.initialize_model(args)
optimizer, scheduler = t.initialize_optimizer_and_scheduler(
    args, model, len(train_loader)
)

t.train(args, model, train_loader, dev_loader, optimizer, scheduler)

model = t.load_model_from_checkpoint(args, best=True)
name = f"t5_scr_{args.experiment_name}_dev"
metrics = t.eval_epoch(
    args, model, dev_loader,
    "data/dev.sql",
    f"results/{name}.sql",
    "records/ground_truth_dev.pkl",
    f"records/{name}.pkl",
)
print("BEST DEV (loss, F1, record EM, SQL EM, error rate):",
      metrics, flush=True)
PY
