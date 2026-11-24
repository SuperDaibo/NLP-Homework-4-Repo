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
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from load_data import get_dataloader
from utils import save_queries_and_records

assert torch.cuda.is_available(), "GPU unavailable"
device = torch.device("cuda")

checkpoint = (
    "checkpoints/scr_experiments/"
    "scratch_continue_lr3e5_24290877/best"
)
print("Checkpoint:", checkpoint, flush=True)

model = T5ForConditionalGeneration.from_pretrained(
    checkpoint
).to(device)
model.eval()

loader = get_dataloader(16, "test")
tokenizer = loader.dataset.tokenizer
queries = []

with torch.no_grad():
    for encoder_ids, encoder_mask, starts in tqdm(
        loader, desc="Generating extra-credit test SQL"
    ):
        generated = model.generate(
            input_ids=encoder_ids.to(device),
            attention_mask=encoder_mask.to(device),
            decoder_input_ids=starts.to(device),
            num_beams=4,
            do_sample=False,
            max_new_tokens=512,
        )
        queries.extend(
            tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
        )

assert len(queries) == len(loader.dataset) == 432

save_queries_and_records(
    queries,
    "results/t5_ft_experiment_ec_test.sql",
    "records/t5_ft_experiment_ec_test.pkl",
)
print("Saved extra-credit outputs for 432 test examples.", flush=True)
PY
