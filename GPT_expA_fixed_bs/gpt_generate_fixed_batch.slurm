#!/bin/bash
#SBATCH --job-name=gpt2-fixed-batch8
#SBATCH --output=gpt_fixed_batch_logs/out_%A_%a.log
#SBATCH --error=gpt_fixed_batch_logs/err_%A_%a.log
#SBATCH --array=1-50%4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00


module load python/3.10
source ~/env/bin/activate
mkdir -p gpt_fixed_batch_logs
# Replace with your actual path
MODEL_PATH="/project/6006459/huggingface_cache/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

srun python gpt_generate_fixed_batch2.py \
    --job_id $SLURM_ARRAY_TASK_ID \
    --output_dir gpt_fixed_batch_logs \
    --batch_size 8 \
    --model_path $MODEL_PATH

