#!/bin/bash
#SBATCH --job-name=gpt2-fized_batch
#SBATCH --output=llama_fixed_batch_logs/out_%A_%a.log
#SBATCH --error=llama_fixed_batch_logs/err_%A_%a.log
#SBATCH --array=1-50%4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00


module load python/3.10
source ~/env/bin/activate
mkdir llama_fixed_batch_logs
# Replace with your actual path
#MODEL_PATH="/project/6006459/huggingface_cache/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_PATH="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
srun python generate_fixed_batch2.py \
    --job_id $SLURM_ARRAY_TASK_ID \
    --output_dir llama_fixed_batch_logs \
    --batch_size 8 \
    --model_path $MODEL_PATH

