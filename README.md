# NLP Assignment 4

## Part 1: BERT Sentiment Classification

- `part1/main.py`: BERT training, evaluation, and data augmentation.
- `part1/utils.py`: Keyboard-typo transformation for movie reviews.

## Part 2: T5 Text-to-SQL

- `part2/load_data.py`: Data preprocessing, tokenization, and batch padding.
- `part2/t5_utils.py`: Model initialization, checkpoint saving/loading, and optimizer setup.
- `part2/train_t5.py`: T5 training and development evaluation.
- `part2/utils.py`: SQL execution and evaluation metrics.
- `part2/evaluate.py`: Evaluation of saved predictions.
- `part2/requirements.txt`: Required Python packages.

## Extra-Credit Job Scripts

- `part2/scratch_t5.sbatch`: Trains T5 from randomly initialized weights.
- `part2/continue_t5_ec.sbatch`: Continues scratch-model training at learning rate `1e-4`.
- `part2/continue_t5_ec_lr3e5.sbatch`: Continues scratch-model training at learning rate `3e-5`.
- `part2/test_t5_ec_final.sbatch`: Generates test SQL and database records from the selected scratch checkpoint.

The job scripts use the NYU Torch Slurm and Apptainer environment.
Model checkpoints are shared through the Google Drive links in the written report.
