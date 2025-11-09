#!/bin/bash
#SBATCH --job-name=send_training
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err

# Try GPU partitions in order of preference
# First try H200 GPUs
if sinfo -p gpu-h200 &>/dev/null; then
    echo "Requesting H200 GPU partition..."
    #SBATCH --partition=gpu-h200
    #SBATCH --gres=gpu:h200:1
elif sinfo -p gpu &>/dev/null; then
    echo "Requesting generic GPU partition..."
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
elif sinfo -p long-gpu &>/dev/null; then
    echo "Requesting long-gpu partition..."
    #SBATCH --partition=long-gpu
    #SBATCH --gres=gpu:1
else
    echo "No GPU partition found, running on CPU..."
fi

# Activate conda environment
source ~/.bashrc
conda activate flsend_clean

# Print environment info
echo "=== Job Environment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=== Starting Training ==="

# Change to working directory
cd ~/FL_SEND/24oct/FL_SEND

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run training
python src/SEND_PSE_AMI.py --test_size 10 --epochs 2 --compute_der_during_training

echo "=== Training Complete ==="

