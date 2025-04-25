#!/bin/bash
#SBATCH --job-name=llama2-batch1to128
#SBATCH --output=llama_logs/llama_batch_out_%A_%a.log
#SBATCH --error=llama_logs/llama_batch_err_%A_%a.log
#SBATCH --array=0-8%4       # 7 batch sizes: index 0 to 6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

module load python/3.10
source ~/env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_PATH="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
#BATCH_SIZES=(1 2 4 8 16 32 64)
BATCH_SIZES=(1 2 4 8 16 32 64 96 128)

BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
mkdir -p llama_logs
echo "[INFO] Running on: $(hostname)"
echo "[INFO] Batch size: $BATCH_SIZE"
nvidia-smi

srun python llama_generate_fixed_token.py \
    --job_id "11B${BATCH_SIZE}" \
    --output_dir llama_logs \
    --batch_size $BATCH_SIZE \
    --model_path $MODEL_PATH

