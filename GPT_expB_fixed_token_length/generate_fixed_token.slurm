#!/bin/bash
#SBATCH --job-name=gpt2-batch
#SBATCH --output=gpt2_fixed_token_logs/out_%A_%a.log
#SBATCH --error=gpt2_fixed_token_logs/err_%A_%a.log
#SBATCH --array=0-6%4       # 7 batch sizes: index 0 to 6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

module load python/3.10
source ~/env/bin/activate

MODEL_PATH="/project/6006459/huggingface_cache/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
BATCH_SIZES=(1 2 4 8 16 32 64)
BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
mkdir -p gpt2_fixed_token_logs
srun python generate_fixed_token.py \
    --job_id "B${BATCH_SIZE}" \
    --output_dir gpt2_fixed_token_logs \
    --batch_size $BATCH_SIZE \
    --model_path $MODEL_PATH

