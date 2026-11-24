#!/bin/bash
#SBATCH --job-name=a4-t5-ec-continue
#SBATCH --account=torch_pr_932_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/a4-t5-ec-continue-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset A4_START_CHECKPOINT

run="scratch_continue_lr1e4_${SLURM_JOB_ID}"

apptainer exec --nv --bind /scratch \
  --overlay "$PWD/a4-python.ext3:ro" \
  /share/apps/images/ubuntu-22.04.4.sif \
  /ext3/venv/bin/python -u - \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --test_batch_size 16 \
    --scheduler_type cosine \
    --num_warmup_epochs 0 \
    --max_n_epochs 10 \
    --patience_epochs 5 \
    --experiment_name "$run" <<'PY'
import os
import torch
from transformers import T5ForConditionalGeneration, set_seed
import train_t5 as t

assert torch.cuda.is_available(), "GPU unavailable"
set_seed(42)

args = t.get_args()
assert not args.finetune

source = (
    "checkpoints/scr_experiments/"
    "scratch_lr3e4_24256483/best"
)
checkpoint_dir = os.path.join(
    "checkpoints", "scr_experiments", args.experiment_name
)
name = f"t5_scr_{args.experiment_name}"

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Starting checkpoint:", source, flush=True)
print("Configuration:", vars(args), flush=True)
print("Output checkpoint directory:", checkpoint_dir, flush=True)

train_loader, dev_loader, _ = t.load_t5_data(
    args.batch_size, args.test_batch_size
)

# Load weights descended from the randomly initialized scratch model.
model = T5ForConditionalGeneration.from_pretrained(
    source
).to(t.DEVICE)

# The continuation uses a new optimizer and scheduler.
optimizer, scheduler = t.initialize_optimizer_and_scheduler(
    args, model, len(train_loader)
)

def evaluate(model, suffix):
    return t.eval_epoch(
        args, model, dev_loader,
        "data/dev.sql",
        f"results/{name}_{suffix}.sql",
        "records/ground_truth_dev.pkl",
        f"records/{name}_{suffix}.pkl",
    )

# Preserve the starting model as the initial best candidate.
baseline = evaluate(model, "baseline_dev")
best_f1 = baseline[1]
t.save_model(checkpoint_dir, model, best=True)
print("BASELINE DEV:", baseline, flush=True)

epochs_without_improvement = 0

for epoch in range(args.max_n_epochs):
    train_loss = t.train_epoch(
        args, model, train_loader, optimizer, scheduler
    )
    print(
        f"Continuation epoch {epoch}: train loss={train_loss}",
        flush=True,
    )

    # Save completed training progress before development evaluation.
    t.save_model(checkpoint_dir, model, best=False)

    metrics = evaluate(model, "dev")
    loss, f1, record_em, sql_em, error_rate = metrics
    print(
        f"Continuation epoch {epoch}: "
        f"Dev loss={loss}, Record F1={f1}, "
        f"Record EM={record_em}, SQL EM={sql_em}, "
        f"Execution error rate={error_rate}",
        flush=True,
    )

    if f1 > best_f1:
        best_f1 = f1
        epochs_without_improvement = 0
        t.save_model(checkpoint_dir, model, best=True)
        print("Saved new best checkpoint.", flush=True)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= args.patience_epochs:
        print("Early stopping.", flush=True)
        break

model = t.load_model_from_checkpoint(args, best=True)
final_metrics = evaluate(model, "best_dev")
print(
    "BEST DEV (loss, F1, record EM, SQL EM, error rate):",
    final_metrics,
    flush=True,
)
print("Best development F1 observed:", best_f1, flush=True)
print("Finished. No test predictions generated.", flush=True)
PY
